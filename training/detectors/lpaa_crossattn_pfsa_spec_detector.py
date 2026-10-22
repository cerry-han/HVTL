"""LPAA CrossAttn-HERL with shallow semantic specificity PFSA."""

import logging

import torch
import torch.nn.functional as F

from detectors import DETECTOR
from .lpaa_crossattn_herl_detector import LPAACrossAttnHERLDetector


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name="lpaa_crossattn_pfsa_spec")
class LPAACrossAttnPFSASpecDetector(LPAACrossAttnHERLDetector):
    """Add the shallow-specificity component of PFSA to LPAA-HERL.

    For a shallow visual level i, the matched text t_i should be more
    compatible than any higher-level text t_j, j > i. This discourages early
    visual layers from collapsing prematurely into global fake semantics.
    """

    def __init__(self, config):
        super().__init__(config)
        self.current_epoch = 0
        self.pfsa_spec_weight = float(config.get("pfsa_spec_weight", 0.05))
        self.pfsa_margin_spec = float(config.get("pfsa_margin_spec", 0.03))
        self.pfsa_start_epoch = int(config.get("pfsa_start_epoch", 1))
        self.pfsa_warmup_epochs = int(config.get("pfsa_warmup_epochs", 2))
        self.pfsa_fake_only = bool(config.get("pfsa_fake_only", True))
        logger.info(
            "[LPAA-PFSA-Spec] weight=%.4f, margin=%.4f, start_epoch=%d, "
            "warmup_epochs=%d, fake_only=%s",
            self.pfsa_spec_weight,
            self.pfsa_margin_spec,
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
            return self.pfsa_spec_weight
        progress = (
            self.current_epoch - self.pfsa_start_epoch + 1
        ) / float(self.pfsa_warmup_epochs)
        return self.pfsa_spec_weight * min(1.0, max(0.0, progress))

    def _encode_pfsa_text(self, input_ids):
        # After the CLIP warmup epoch the text encoder is frozen. Avoid
        # retaining its graph while preserving gradients to visual adapters.
        if self.current_epoch >= self.clip_warmup_epochs:
            with torch.no_grad():
                return self.encode_text(input_ids)
        return self.encode_text(input_ids)

    def _compute_pfsa_spec_loss(self, data_dict, pred_dict):
        label = data_dict["label"]
        positive_by_layer = data_dict.get("text_tokens", {})
        layers = list(self.adapter_layers)
        reference = pred_dict[f"v_proj_{layers[0]}"]
        device = reference.device

        total_loss = reference.new_tensor(0.0)
        order_violation_count = reference.new_tensor(0.0)
        margin_violation_count = reference.new_tensor(0.0)
        constraint_count = 0

        # L3 is compared with L6/L9/L12 text, L6 with L9/L12 text, and
        # L9 with L12 text. L12 has no higher abstraction level to suppress.
        for shallow_position in range(len(layers) - 1):
            shallow_layer = layers[shallow_position]
            matched_tokens = positive_by_layer.get(f"L{shallow_layer}")
            if matched_tokens is None:
                continue

            for high_position in range(shallow_position + 1, len(layers)):
                high_layer = layers[high_position]
                high_tokens = positive_by_layer.get(f"L{high_layer}")
                if high_tokens is None:
                    continue

                valid_mask = (
                    (matched_tokens.sum(dim=-1) > 0)
                    & (high_tokens.sum(dim=-1) > 0)
                ).to(label.device)
                if self.pfsa_fake_only:
                    valid_mask = valid_mask & (label == 1)
                if not torch.any(valid_mask):
                    continue

                token_mask = valid_mask.to(matched_tokens.device)
                visual_mask = valid_mask.to(device)
                all_tokens = torch.cat(
                    [matched_tokens[token_mask], high_tokens[token_mask]],
                    dim=0,
                )
                matched_text, high_text = self._encode_pfsa_text(
                    all_tokens
                ).chunk(2, dim=0)
                visual = pred_dict[f"v_proj_{shallow_layer}"][visual_mask]

                matched_similarity = torch.sum(
                    visual * matched_text, dim=-1
                )
                high_similarity = torch.sum(visual * high_text, dim=-1)
                gap = matched_similarity - high_similarity

                total_loss = total_loss + F.relu(
                    self.pfsa_margin_spec - gap
                ).sum()
                order_violation_count = order_violation_count + (
                    gap <= 0
                ).float().sum()
                margin_violation_count = margin_violation_count + (
                    gap < self.pfsa_margin_spec
                ).float().sum()
                constraint_count += int(gap.numel())

        if constraint_count == 0:
            zero = reference.new_tensor(0.0)
            return zero, zero, zero, zero

        count = reference.new_tensor(float(constraint_count))
        spec_loss = total_loss / count
        order_violation_rate = order_violation_count / count
        margin_violation_rate = margin_violation_count / count
        return spec_loss, order_violation_rate, margin_violation_rate, count

    def get_losses(self, data_dict, pred_dict):
        loss_dict = super().get_losses(data_dict, pred_dict)
        base_loss = loss_dict.get("overall")
        if base_loss is None:
            base_loss = data_dict["label"].new_tensor(
                0.0, dtype=torch.float32
            )

        # PFSA is a training-only auxiliary objective. Test predictions do not
        # expose v_proj_* features, so return the original test loss unchanged.
        missing_pfsa_visual = any(
            f"v_proj_{layer}" not in pred_dict for layer in self.adapter_layers
        )
        if missing_pfsa_visual:
            zero = base_loss.new_tensor(0.0)
            loss_dict["pfsa_spec"] = zero
            loss_dict["pfsa_spec_order_vr"] = zero
            loss_dict["pfsa_spec_margin_vr"] = zero
            loss_dict["pfsa_spec_constraints"] = zero
            loss_dict["pfsa_spec_weight_active"] = zero
            return loss_dict

        active_weight = self._active_pfsa_weight()
        reference = pred_dict[f"v_proj_{self.adapter_layers[0]}"]
        if active_weight <= 0.0:
            zero = reference.new_tensor(0.0)
            loss_dict["pfsa_spec"] = zero
            loss_dict["pfsa_spec_order_vr"] = zero
            loss_dict["pfsa_spec_margin_vr"] = zero
            loss_dict["pfsa_spec_constraints"] = zero
            loss_dict["pfsa_spec_weight_active"] = zero
            return loss_dict

        spec_loss, order_vr, margin_vr, constraint_count = (
            self._compute_pfsa_spec_loss(data_dict, pred_dict)
        )
        active_weight_tensor = reference.new_tensor(float(active_weight))
        loss_dict["pfsa_spec"] = spec_loss
        loss_dict["pfsa_spec_order_vr"] = order_vr
        loss_dict["pfsa_spec_margin_vr"] = margin_vr
        loss_dict["pfsa_spec_constraints"] = constraint_count
        loss_dict["pfsa_spec_weight_active"] = active_weight_tensor
        loss_dict["overall"] = (
            loss_dict["overall"] + active_weight_tensor * spec_loss
        )
        return loss_dict
