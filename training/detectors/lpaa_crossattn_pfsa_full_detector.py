"""LPAA CrossAttn-HERL with the complete PFSA objective."""

import logging

import torch
import torch.nn.functional as F

from detectors import DETECTOR
from .lpaa_crossattn_pfsa_detector import LPAACrossAttnPFSADetector


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name="lpaa_crossattn_pfsa_full")
class LPAACrossAttnPFSAFullDetector(LPAACrossAttnPFSADetector):
    """Combine monotonic acquisition and shallow specificity constraints."""

    def __init__(self, config):
        super().__init__(config)
        self.pfsa_margin_spec = float(config.get("pfsa_margin_spec", 0.03))
        self.pfsa_spec_scale = float(config.get("pfsa_spec_scale", 0.25))
        logger.info(
            "[LPAA-PFSA-Full] rho=%.4f, mono_margin=%.4f, "
            "spec_margin=%.4f, spec_scale=%.4f",
            self.pfsa_weight,
            self.pfsa_margin_mono,
            self.pfsa_margin_spec,
            self.pfsa_spec_scale,
        )

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

        # A shallow visual level should prefer its matched text over every
        # available higher-level text. L12 is allowed to retain lower-level
        # evidence and therefore has no specificity constraint.
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
        return (
            total_loss / count,
            order_violation_count / count,
            margin_violation_count / count,
            count,
        )

    def get_losses(self, data_dict, pred_dict):
        # The parent adds rho * L_mono to the original LPAA-HERL objective.
        loss_dict = super().get_losses(data_dict, pred_dict)
        base_loss = loss_dict.get("overall")
        if base_loss is None:
            base_loss = data_dict["label"].new_tensor(
                0.0, dtype=torch.float32
            )

        missing_visual = any(
            f"v_proj_{layer}" not in pred_dict for layer in self.adapter_layers
        )
        if missing_visual:
            zero = base_loss.new_tensor(0.0)
            loss_dict["pfsa_spec"] = zero
            loss_dict["pfsa_spec_order_vr"] = zero
            loss_dict["pfsa_spec_margin_vr"] = zero
            loss_dict["pfsa_spec_constraints"] = zero
            loss_dict["pfsa_spec_scale"] = zero
            loss_dict["pfsa_full"] = loss_dict.get("pfsa_mono", zero)
            return loss_dict

        active_weight = self._active_pfsa_weight()
        reference = pred_dict[f"v_proj_{self.adapter_layers[0]}"]
        if active_weight <= 0.0 or self.pfsa_spec_scale <= 0.0:
            zero = reference.new_tensor(0.0)
            loss_dict["pfsa_spec"] = zero
            loss_dict["pfsa_spec_order_vr"] = zero
            loss_dict["pfsa_spec_margin_vr"] = zero
            loss_dict["pfsa_spec_constraints"] = zero
            loss_dict["pfsa_spec_scale"] = reference.new_tensor(
                self.pfsa_spec_scale
            )
            loss_dict["pfsa_full"] = loss_dict.get("pfsa_mono", zero)
            return loss_dict

        spec_loss, order_vr, margin_vr, constraint_count = (
            self._compute_pfsa_spec_loss(data_dict, pred_dict)
        )
        active_weight_tensor = reference.new_tensor(float(active_weight))
        spec_scale_tensor = reference.new_tensor(self.pfsa_spec_scale)
        mono_loss = loss_dict.get("pfsa_mono", reference.new_tensor(0.0))

        loss_dict["pfsa_spec"] = spec_loss
        loss_dict["pfsa_spec_order_vr"] = order_vr
        loss_dict["pfsa_spec_margin_vr"] = margin_vr
        loss_dict["pfsa_spec_constraints"] = constraint_count
        loss_dict["pfsa_spec_scale"] = spec_scale_tensor
        loss_dict["pfsa_full"] = mono_loss + spec_scale_tensor * spec_loss
        loss_dict["overall"] = (
            loss_dict["overall"]
            + active_weight_tensor * spec_scale_tensor * spec_loss
        )
        return loss_dict
