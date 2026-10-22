"""Controlled layer-weight ablations for the final LPAA-HERL-PFSA model."""

import logging

from detectors import DETECTOR
from .lpaa_crossattn_pfsa_detector import LPAACrossAttnPFSADetector


logger = logging.getLogger(__name__)


DEFAULT_PROGRESSIVE_SCHEDULE = [
    {"until": 0.25, "weights": {3: 1.0, 6: 0.3, 9: 0.2, 12: 0.1}},
    {"until": 0.50, "weights": {3: 0.5, 6: 1.0, 9: 0.4, 12: 0.2}},
    {"until": 0.75, "weights": {3: 0.3, 6: 0.5, 9: 1.0, 12: 0.5}},
    {"until": 1.00, "weights": {3: 0.2, 6: 0.3, 9: 0.5, 12: 1.2}},
]


def _normalize_weights(weights):
    return {int(layer): float(weight) for layer, weight in weights.items()}


@DETECTOR.register_module(module_name="lpaa_crossattn_pfsa_lambda_ablation")
class LPAACrossAttnPFSALambdaAblationDetector(LPAACrossAttnPFSADetector):
    """Apply an explicit progressive or fixed layer-loss schedule."""

    def __init__(self, config):
        super().__init__(config)
        self.lambda_ablation_mode = str(
            config.get("lambda_ablation_mode", "progressive")
        ).lower()
        self.total_lambda_epochs = int(config.get("nEpochs", 10))
        self.fixed_lambda_weights = _normalize_weights(
            config.get("lambda_weights", {3: 0.5, 6: 0.5, 9: 0.5, 12: 0.5})
        )
        self.progressive_lambda_schedule = config.get(
            "progressive_lambda_schedule", DEFAULT_PROGRESSIVE_SCHEDULE
        )

        valid_modes = {"progressive", "fixed_equal", "fixed_deep"}
        if self.lambda_ablation_mode not in valid_modes:
            raise ValueError(
                "lambda_ablation_mode must be one of "
                f"{sorted(valid_modes)}, got {self.lambda_ablation_mode!r}"
            )
        self._apply_lambda_schedule(epoch=0)

    def _progressive_weights(self, epoch):
        denominator = max(float(self.total_lambda_epochs - 1), 1.0)
        progress = min(max(float(epoch) / denominator, 0.0), 1.0)
        for stage in self.progressive_lambda_schedule:
            if progress <= float(stage.get("until", 1.0)):
                return _normalize_weights(stage["weights"])
        return _normalize_weights(self.progressive_lambda_schedule[-1]["weights"])

    def _apply_lambda_schedule(self, epoch):
        if self.lambda_ablation_mode == "progressive":
            weights = self._progressive_weights(epoch)
        else:
            weights = self.fixed_lambda_weights
        self.lambda_weights = dict(weights)
        logger.info(
            "[LPAA-LambdaAblation] epoch=%d mode=%s weights=%s",
            int(epoch),
            self.lambda_ablation_mode,
            self.lambda_weights,
        )

    def on_epoch_start(self, epoch):
        # Keep the existing CLIP warmup/freeze and PFSA behavior unchanged.
        super().on_epoch_start(epoch)
        # Apply the controlled layer weights last, even if a parent class also
        # contains a progressive schedule.
        self._apply_lambda_schedule(epoch)
