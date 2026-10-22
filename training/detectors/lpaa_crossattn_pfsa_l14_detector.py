"""ViT-L/14 variant of the final LPAA-HERL-PFSA detector."""

import logging

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from detectors import DETECTOR
from .lpaa_crossattn_pfsa_detector import LPAACrossAttnPFSADetector


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name="lpaa_crossattn_pfsa_l14")
class LPAACrossAttnPFSAL14Detector(LPAACrossAttnPFSADetector):
    """Run the final LPAA objective with a CLIP ViT-L/14 visual backbone.

    The dataset keeps the four semantic text levels named L3/L6/L9/L12.
    ViT-L/14 has 24 visual blocks, so those semantic levels are sampled from
    physical blocks 6/12/18/24 respectively.
    """

    HIDDEN_DIM = 1024

    def __init__(self, config):
        config = dict(config)
        config["clip_model_name"] = config.get(
            "clip_model_name", "openai/clip-vit-large-patch14"
        )
        config["adapter_layers"] = [3, 6, 9, 12]
        config["evidence_layers"] = [3, 6, 9]
        config["query_layer"] = 12

        super().__init__(config)

        capture_map = config.get(
            "l14_capture_map", {6: 3, 12: 6, 18: 9, 24: 12}
        )
        self.capture_to_semantic = {
            int(block): int(level) for block, level in capture_map.items()
        }

        vision_hidden = int(self.backbone.config.hidden_size)
        if vision_hidden != self.HIDDEN_DIM:
            raise RuntimeError(
                "Expected ViT-L/14 hidden size 1024, got "
                f"{vision_hidden} from {config['clip_model_name']}"
            )
        block_count = len(self.backbone.encoder.layers)
        if max(self.capture_to_semantic) > block_count:
            raise RuntimeError(
                f"Capture block exceeds backbone depth {block_count}: "
                f"{self.capture_to_semantic}"
            )
        if set(self.capture_to_semantic.values()) != set(self.adapter_layers):
            raise RuntimeError(
                "l14_capture_map values must match semantic adapter layers: "
                f"{self.adapter_layers}"
            )

        projection_dim = getattr(self.text_projection, "out_features", None)
        if projection_dim is None:
            projection_dim = int(self.text_projection.weight.shape[0])
        self.v_projections = nn.ModuleDict(
            {
                str(layer): nn.Linear(self.HIDDEN_DIM, projection_dim)
                for layer in self.adapter_layers
            }
        )

        self.use_activation_checkpointing = bool(
            config.get("gradient_checkpointing", True)
        )
        if self.use_activation_checkpointing:
            # LPAACrossAttnDetector walks through vision blocks manually, so
            # PreTrainedModel.gradient_checkpointing_enable() does not cover
            # that path. Vision blocks are checkpointed explicitly in
            # features(). CLIPTextTransformer is not a PreTrainedModel either;
            # configure its nested CLIPEncoder exactly as CLIPModel would.
            text_encoder = getattr(self.text_encoder, "encoder", None)
            if text_encoder is None:
                raise RuntimeError("CLIP text encoder has no encoder module")
            text_encoder.gradient_checkpointing = True
            text_encoder._gradient_checkpointing_func = (
                self._checkpoint_text_layer
            )

        logger.info(
            "[LPAA-PFSA-L14] hidden=%d, projection=%d, blocks=%d, "
            "capture_map=%s",
            self.HIDDEN_DIM,
            projection_dim,
            block_count,
            self.capture_to_semantic,
        )

    @staticmethod
    def _checkpoint_text_layer(function, *args):
        return checkpoint(function, *args, use_reentrant=False)

    def features(self, data_dict):
        hidden = self.backbone.embeddings(data_dict["image"])
        hidden = self.backbone.pre_layrnorm(hidden)

        adapter_outputs = {}
        for block_index, layer in enumerate(
            self.backbone.encoder.layers, start=1
        ):
            if (
                self.use_activation_checkpointing
                and self.training
                and hidden.requires_grad
            ):
                def forward_block(states, current_layer=layer):
                    return current_layer(
                        states,
                        attention_mask=None,
                        causal_attention_mask=None,
                    )[0]

                hidden = checkpoint(
                    forward_block, hidden, use_reentrant=False
                )
            else:
                layer_out = layer(
                    hidden,
                    attention_mask=None,
                    causal_attention_mask=None,
                )
                hidden = layer_out[0]
            semantic_level = self.capture_to_semantic.get(block_index)
            if semantic_level is not None:
                key = str(semantic_level)
                adapter_outputs[key] = self.adapters[key](
                    hidden[:, 1:, :], hidden[:, 0, :]
                )

        missing = [
            layer
            for layer in self.adapter_layers
            if str(layer) not in adapter_outputs
        ]
        if missing:
            raise RuntimeError(
                f"Missing ViT-L/14 semantic adapter outputs: {missing}"
            )
        return adapter_outputs
