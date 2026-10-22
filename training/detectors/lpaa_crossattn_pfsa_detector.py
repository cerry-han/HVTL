"""LPAA CrossAttn-HERL with Progressive Forgery Semantic Abstraction."""

import logging

import torch
import torch.nn.functional as F

from detectors import DETECTOR
from .lpaa_crossattn_herl_detector import LPAACrossAttnHERLDetector


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name="lpaa_crossattn_pfsa")
class LPAACrossAttnPFSADetector(LPAACrossAttnHERLDetector):
    """Add monotonic cross-depth semantic abstraction to LPAA-HERL.

    PFSA is intentionally added after the HERL loss. The default setting keeps
    PFSA inactive during epoch 0, when the full CLIP model is unfrozen, and
    gradually enables it after CLIP is frozen again.
    """

    def __init__(self, config):
        super().__init__(config)
        self.current_epoch = 0
        self.pfsa_weight = float(config.get("pfsa_weight", 0.05))
        self.pfsa_margin_mono = float(config.get("pfsa_margin_mono", 0.03))
        self.pfsa_start_epoch = int(config.get("pfsa_start_epoch", 1))
        self.pfsa_warmup_epochs = int(config.get("pfsa_warmup_epochs", 2))
        self.pfsa_fake_only = bool(config.get("pfsa_fake_only", True))
        logger.info(
            "[LPAA-PFSA] mono_weight=%.4f, margin=%.4f, start_epoch=%d, "
            "warmup_epochs=%d, fake_only=%s",
            self.pfsa_weight,
            self.pfsa_margin_mono,
            self.pfsa_start_epoch,
            self.pfsa_warmup_epochs,
            self.pfsa_fake_only,
        )

    def on_epoch_start(self, epoch):
        super().on_epoch_start(epoch)
        self.current_epoch = int(epoch)

    def _active_pfsa_weight(self):
        if self.current_epoch < self.pfsa_start_epoch:
            return 0.0
        if self.pfsa_warmup_epochs <= 0:
            return self.pfsa_weight
        progress = (
            self.current_epoch - self.pfsa_start_epoch + 1
        ) / float(self.pfsa_warmup_epochs)
        return self.pfsa_weight * min(1.0, max(0.0, progress))

    def _encode_pfsa_text(self, input_ids):
        # PFSA is normally enabled after CLIP is frozen, so disabling text
        # gradients here reduces memory without changing the visual objective.
        if self.current_epoch >= self.clip_warmup_epochs:
            with torch.no_grad():
                return self.encode_text(input_ids)
        return self.encode_text(input_ids)

    def _compute_pfsa_mono_loss(self, data_dict, pred_dict):
        label = data_dict["label"]
        positive_by_layer = data_dict.get("text_tokens", {})
        layers = list(self.adapter_layers)
        reference = pred_dict[f"v_proj_{layers[0]}"]
        device = reference.device

        total_loss = reference.new_tensor(0.0)
        violation_count = reference.new_tensor(0.0)
        constraint_count = 0

        for text_position in range(1, len(layers)):
            text_layer = layers[text_position]
            text_key = f"L{text_layer}"
            tokens = positive_by_layer.get(text_key)
            if tokens is None:
                continue

            valid_mask = (tokens.sum(dim=-1) > 0).to(label.device)
            if self.pfsa_fake_only:
                valid_mask = valid_mask & (label == 1)
            if not torch.any(valid_mask):
                continue

            token_mask = valid_mask.to(tokens.device)
            visual_mask = valid_mask.to(device)
            text_feature = self._encode_pfsa_text(tokens[token_mask])

            for pair_position in range(text_position):
                shallow_layer = layers[pair_position]
                deep_layer = layers[pair_position + 1]
                shallow_visual = pred_dict[
                    f"v_proj_{shallow_layer}"
                ][visual_mask]
                deep_visual = pred_dict[f"v_proj_{deep_layer}"][visual_mask]

                shallow_similarity = torch.sum(
                    shallow_visual * text_feature, dim=-1
                )
                deep_similarity = torch.sum(
                    deep_visual * text_feature, dim=-1
                )
                gap = deep_similarity - shallow_similarity
                total_loss = total_loss + F.relu(
                    self.pfsa_margin_mono - gap
                ).sum()
                violation_count = violation_count + (gap <= 0).float().sum()
                constraint_count += int(gap.numel())

        if constraint_count == 0:
            zero = reference.new_tensor(0.0)
            return zero, zero, zero

        count = reference.new_tensor(float(constraint_count))
        mono_loss = total_loss / count
        violation_rate = violation_count / count
        return mono_loss, violation_rate, count

    def get_losses(self, data_dict, pred_dict):
        loss_dict = super().get_losses(data_dict, pred_dict)
        base_loss = loss_dict.get("overall")
        if base_loss is None:
            base_loss = data_dict["label"].new_tensor(0.0, dtype=torch.float32)

        missing_pfsa_visual = any(
            f"v_proj_{layer}" not in pred_dict for layer in self.adapter_layers
        )
        if missing_pfsa_visual:
            zero = base_loss.new_tensor(0.0)
            loss_dict["pfsa_mono"] = zero
            loss_dict["pfsa_mvr"] = zero
            loss_dict["pfsa_constraints"] = zero
            loss_dict["pfsa_weight_active"] = zero
            return loss_dict

        active_weight = self._active_pfsa_weight()
        reference = pred_dict[f"v_proj_{self.adapter_layers[0]}"]

        if active_weight <= 0.0:
            zero = reference.new_tensor(0.0)
            loss_dict["pfsa_mono"] = zero
            loss_dict["pfsa_mvr"] = zero
            loss_dict["pfsa_constraints"] = zero
            loss_dict["pfsa_weight_active"] = zero
            return loss_dict

        mono_loss, violation_rate, constraint_count = (
            self._compute_pfsa_mono_loss(data_dict, pred_dict)
        )
        active_weight_tensor = reference.new_tensor(float(active_weight))
        loss_dict["pfsa_mono"] = mono_loss
        loss_dict["pfsa_mvr"] = violation_rate
        loss_dict["pfsa_constraints"] = constraint_count
        loss_dict["pfsa_weight_active"] = active_weight_tensor
        loss_dict["overall"] = (
            loss_dict["overall"] + active_weight_tensor * mono_loss
        )
        return loss_dict
