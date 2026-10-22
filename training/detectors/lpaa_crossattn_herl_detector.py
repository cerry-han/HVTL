"""LPAA CrossAttn with Hierarchical Evidence Ranking Loss (HERL)."""

import logging

import torch
import torch.nn.functional as F

from detectors import DETECTOR
from .lpaa_crossattn_detector import LPAACrossAttnDetector


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name="lpaa_crossattn_herl")
class LPAACrossAttnHERLDetector(LPAACrossAttnDetector):
    """Rank matched evidence above hard evidence above natural evidence."""

    def __init__(self, config):
        super().__init__(config)
        self.clip_warmup_epochs = int(config.get("clip_warmup_epochs", 1))
        self.clip_learning_rate = float(
            config.get("clip_learning_rate", 1e-5)
        )
        self.adapter_learning_rate = float(
            config.get("adapter_learning_rate", 1e-4)
        )
        self.herl_weight = float(config.get("herl_weight", 0.2))
        self.herl_margin_ph = float(
            config.get("herl_margin_positive_hard", 0.05)
        )
        self.herl_margin_hn = float(
            config.get("herl_margin_hard_negative", 0.10)
        )
        self.herl_hn_weight = float(
            config.get("herl_hard_negative_weight", 1.0)
        )
        self.herl_use_ph = bool(config.get("herl_use_ph", True))
        self.herl_use_hn = bool(config.get("herl_use_hn", True))
        self.herl_hard_selection = str(
            config.get("herl_hard_selection", "semantic")
        ).strip().lower()
        if self.herl_hard_selection not in {"semantic", "random"}:
            raise ValueError(
                "herl_hard_selection must be semantic or random"
            )
        if not self.herl_use_ph and not self.herl_use_hn:
            raise ValueError(
                "At least one HERL term must be enabled: "
                "herl_use_ph or herl_use_hn"
            )

        # The optimizer is created after detector initialization. Unfreeze the
        # complete CLIP model here so all CLIP parameters enter its LR group.
        self._set_clip_trainable(self.clip_warmup_epochs > 0)
        logger.info(
            "[LPAA-HERL] weight=%.4f, margin_ph=%.4f, "
            "margin_hn=%.4f, hn_weight=%.4f, use_ph=%s, use_hn=%s",
            self.herl_weight,
            self.herl_margin_ph,
            self.herl_margin_hn,
            self.herl_hn_weight,
            self.herl_use_ph,
            self.herl_use_hn,
        )
        logger.info(
            "[LPAA-HERL] CLIP warmup epochs=%d, clip_lr=%.2e, "
            "adapter_lr=%.2e",
            self.clip_warmup_epochs,
            self.clip_learning_rate,
            self.adapter_learning_rate,
        )

    def _clip_modules(self):
        return (
            self.backbone,
            self.text_encoder,
            self.text_projection,
        )

    def _set_clip_trainable(self, trainable):
        for module in self._clip_modules():
            for parameter in module.parameters():
                parameter.requires_grad = bool(trainable)
        logger.info(
            "[LPAA-HERL] full CLIP vision/text model %s",
            "unfrozen" if trainable else "frozen",
        )

    def on_epoch_start(self, epoch):
        parent_hook = getattr(super(), "on_epoch_start", None)
        if callable(parent_hook):
            parent_hook(epoch)
        self._set_clip_trainable(int(epoch) < self.clip_warmup_epochs)

    def encode_text(self, input_ids):
        """Encode text with gradients only while full CLIP is unfrozen."""
        device = next(self.head.parameters()).device
        self.text_encoder = self.text_encoder.to(device)
        self.text_projection = self.text_projection.to(device)
        input_ids = input_ids.to(device)
        output = self.text_encoder(input_ids=input_ids)
        projected = self.text_projection(output.pooler_output)
        return F.normalize(projected, dim=-1)

    def get_optimizer_param_groups(self):
        """Return distinct LR groups for CLIP and task-specific modules."""
        clip_parameters = [
            parameter
            for module in self._clip_modules()
            for parameter in module.parameters()
        ]
        clip_parameter_ids = {id(parameter) for parameter in clip_parameters}
        adapter_parameters = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in clip_parameter_ids
            and parameter.requires_grad
        ]
        if not clip_parameters:
            raise RuntimeError("CLIP vision backbone has no parameters")
        if not adapter_parameters:
            raise RuntimeError("No trainable adapter/head parameters found")

        logger.info(
            "[LPAA-HERL] optimizer groups: CLIP=%d params at %.2e, "
            "adapter/head=%d params at %.2e",
            sum(parameter.numel() for parameter in clip_parameters),
            self.clip_learning_rate,
            sum(parameter.numel() for parameter in adapter_parameters),
            self.adapter_learning_rate,
        )
        return [
            {
                "params": clip_parameters,
                "lr": self.clip_learning_rate,
                "group_name": "full_clip",
            },
            {
                "params": adapter_parameters,
                "lr": self.adapter_learning_rate,
                "group_name": "adapter_head",
            },
        ]

    def _layer_rank_loss(
        self,
        visual,
        positive_tokens,
        hard_candidate_tokens,
        negative_tokens,
        valid_mask,
    ):
        if not torch.any(valid_mask):
            zero = visual.new_tensor(0.0)
            return zero, zero, zero

        visual_mask = valid_mask.to(visual.device)
        token_mask = valid_mask.to(positive_tokens.device)
        visual = visual[visual_mask]
        positive_tokens = positive_tokens[token_mask]
        hard_candidate_tokens = hard_candidate_tokens[token_mask]
        if self.herl_use_hn:
            negative_tokens = negative_tokens[token_mask]

        batch_size, candidate_count, token_length = hard_candidate_tokens.shape
        token_groups = [
            positive_tokens,
            hard_candidate_tokens.reshape(-1, token_length),
        ]
        if self.herl_use_hn:
            token_groups.append(negative_tokens)
        all_text = self.encode_text(torch.cat(token_groups, dim=0))

        positive = all_text[:batch_size]
        hard_start = batch_size
        hard_end = hard_start + batch_size * candidate_count
        hard_candidates = all_text[hard_start:hard_end].reshape(
            batch_size, candidate_count, -1
        )
        negative = all_text[hard_end:] if self.herl_use_hn else None

        if self.herl_hard_selection == "semantic":
            semantic_similarity = torch.sum(
                hard_candidates * positive.unsqueeze(1), dim=-1
            )
            selected_index = semantic_similarity.argmax(dim=1)
        else:
            # Fixed global seeds make this random-candidate control
            # reproducible while avoiding semantic-nearest mining.
            selected_index = torch.randint(
                candidate_count,
                (batch_size,),
                device=hard_candidates.device,
            )
        gather_index = selected_index.view(-1, 1, 1).expand(
            -1, 1, hard_candidates.shape[-1]
        )
        hard = hard_candidates.gather(1, gather_index).squeeze(1)

        positive_similarity = torch.sum(visual * positive, dim=-1)
        hard_similarity = torch.sum(visual * hard, dim=-1)
        zero = visual.new_tensor(0.0)

        if self.herl_use_ph:
            positive_hard_loss = F.relu(
                self.herl_margin_ph
                - positive_similarity
                + hard_similarity
            ).mean()
        else:
            positive_hard_loss = zero

        if self.herl_use_hn:
            negative_similarity = torch.sum(visual * negative, dim=-1)
            hard_negative_loss = F.relu(
                self.herl_margin_hn
                - hard_similarity
                + negative_similarity
            ).mean()
        else:
            hard_negative_loss = zero

        rank_loss = (
            positive_hard_loss
            + self.herl_hn_weight * hard_negative_loss
        )
        return rank_loss, positive_hard_loss, hard_negative_loss

    def get_losses(self, data_dict, pred_dict):
        loss_dict = super().get_losses(data_dict, pred_dict)
        label = data_dict["label"]
        positive_by_layer = data_dict.get("text_tokens", {})
        rank_tokens = data_dict.get("evidence_rank_tokens", {})
        hard_by_layer = rank_tokens.get("hard_candidates", {})
        negative_by_layer = rank_tokens.get("natural_negative", {})

        total_rank_loss = label.new_tensor(0.0, dtype=torch.float32)
        for layer in self.adapter_layers:
            key = f"L{layer}"
            positive_tokens = positive_by_layer.get(key)
            hard_tokens = hard_by_layer.get(key)
            negative_tokens = negative_by_layer.get(key)
            missing_required_tokens = (
                positive_tokens is None
                or hard_tokens is None
                or negative_tokens is None
            )
            if missing_required_tokens:
                zero = total_rank_loss.new_tensor(0.0)
                loss_dict[f"herl_ph_L{layer}"] = zero
                loss_dict[f"herl_hn_L{layer}"] = zero
                loss_dict[f"herl_L{layer}"] = zero
                continue

            valid_mask = (
                (positive_tokens.sum(dim=-1) > 0).to(label.device)
                & (hard_tokens.sum(dim=-1) > 0)
                .any(dim=-1)
                .to(label.device)
                & (negative_tokens.sum(dim=-1) > 0).to(label.device)
                & (label == 1)
            )

            layer_loss, ph_loss, hn_loss = self._layer_rank_loss(
                pred_dict[f"v_proj_{layer}"],
                positive_tokens,
                hard_tokens,
                negative_tokens,
                valid_mask,
            )
            loss_dict[f"herl_ph_L{layer}"] = ph_loss
            loss_dict[f"herl_hn_L{layer}"] = hn_loss
            loss_dict[f"herl_L{layer}"] = layer_loss

            layer_weight = float(self.lambda_weights.get(layer, 0.5))
            total_rank_loss = total_rank_loss + layer_weight * layer_loss

        loss_dict["herl_rank"] = total_rank_loss
        loss_dict["overall"] = (
            loss_dict["overall"] + self.herl_weight * total_rank_loss
        )
        return loss_dict
