"""
LPAA-CrossAttn detector.

This variant keeps the original LPAA evidence adapters at layers 3/6/9/12,
but changes the final decision path: layer 12 acts as a semantic query and
cross-attends to evidence tokens from layers 3/6/9 before classification.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from .lpaa_detector import HorizontalAdapter, LayerAlignLoss
from detectors import DETECTOR
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='lpaa_crossattn')
class LPAACrossAttnDetector(AbstractDetector):
    HIDDEN_DIM = 768

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.adapter_layers = config.get('adapter_layers', [3, 6, 9, 12])
        self.evidence_layers = config.get('evidence_layers', [3, 6, 9])
        self.query_layer = config.get('query_layer', 12)
        self.temperature = config.get('temperature', 0.07)
        self.beta = config.get('beta', 1.0)
        self.base_lambda_weights = config.get(
            'lambda_weights',
            {3: 0.2, 6: 0.2, 9: 0.2, 12: 1.0},
        )
        self.lambda_weights = dict(self.base_lambda_weights)
        self.use_progressive_lambda = config.get('use_progressive_lambda', False)
        self.progressive_lambda_schedule = config.get(
            'progressive_lambda_schedule',
            [
                {'until': 0.25, 'weights': {3: 1.0, 6: 0.3, 9: 0.2, 12: 0.1}},
                {'until': 0.50, 'weights': {3: 0.5, 6: 1.0, 9: 0.4, 12: 0.2}},
                {'until': 0.75, 'weights': {3: 0.3, 6: 0.5, 9: 1.0, 12: 0.5}},
                {'until': 1.00, 'weights': {3: 0.2, 6: 0.3, 9: 0.5, 12: 1.2}},
            ],
        )
        self.current_epoch = 0
        self.total_epochs = config.get('nEpochs', 20)
        bottleneck = config.get('adapter_bottleneck', 64)
        dropout = config.get('dropout', 0.1)
        clip_name = config.get('clip_model_name', 'openai/clip-vit-base-patch16')

        self.backbone = self.build_backbone(config)

        self.adapters = nn.ModuleDict({
            str(layer): HorizontalAdapter(self.HIDDEN_DIM, bottleneck, dropout)
            for layer in self.adapter_layers
        })

        self.v_projections = nn.ModuleDict({
            str(layer): nn.Linear(self.HIDDEN_DIM, 512)
            for layer in self.adapter_layers
        })

        num_heads = config.get('cross_attn_heads', 8)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.HIDDEN_DIM,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.HIDDEN_DIM)
        self.cross_attn_dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

        self.cls_loss_func = self.build_loss(config)
        self.align_loss_func = LayerAlignLoss(self.temperature)

        clip_model = CLIPModel.from_pretrained(clip_name)
        self.text_encoder = clip_model.text_model
        self.text_projection = clip_model.text_projection
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.text_projection.parameters():
            p.requires_grad = False

        logger.info(
            f"[LPAA-CrossAttn] evidence_layers={self.evidence_layers}, "
            f"query_layer={self.query_layer}, lambda={self.lambda_weights}, beta={self.beta}"
        )

    def build_backbone(self, config):
        clip_name = config.get('clip_model_name', 'openai/clip-vit-base-patch16')
        vision = CLIPModel.from_pretrained(clip_name).vision_model
        for p in vision.parameters():
            p.requires_grad = False
        logger.info("[LPAA-CrossAttn] frozen CLIP vision backbone")
        return vision

    def on_epoch_start(self, epoch: int):
        self.current_epoch = epoch
        self.update_progressive_lambda(epoch)

    def update_progressive_lambda(self, epoch: int):
        if not self.use_progressive_lambda:
            self.lambda_weights = dict(self.base_lambda_weights)
            return

        denom = max(float(self.total_epochs - 1), 1.0)
        progress = min(max(float(epoch) / denom, 0.0), 1.0)

        for stage in self.progressive_lambda_schedule:
            if progress <= float(stage.get('until', 1.0)):
                weights = stage.get('weights', self.base_lambda_weights)
                self.lambda_weights = {int(k): float(v) for k, v in weights.items()}
                return

        weights = self.progressive_lambda_schedule[-1].get('weights', self.base_lambda_weights)
        self.lambda_weights = {int(k): float(v) for k, v in weights.items()}

    def build_loss(self, config):
        return LOSSFUNC[config.get('loss_func', 'cross_entropy')]()

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = next(self.head.parameters()).device
        self.text_encoder = self.text_encoder.to(device)
        self.text_projection = self.text_projection.to(device)
        input_ids = input_ids.to(device)
        out = self.text_encoder(input_ids=input_ids)
        projected = self.text_projection(out.pooler_output)
        return F.normalize(projected, dim=-1)

    def features(self, data_dict: dict) -> dict:
        hidden = self.backbone.embeddings(data_dict['image'])
        hidden = self.backbone.pre_layrnorm(hidden)

        attention_mask = None
        causal_attention_mask = None

        adapter_outputs = {}
        for idx, layer in enumerate(self.backbone.encoder.layers, start=1):
            layer_out = layer(
                hidden,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
            )
            hidden = layer_out[0]
            if idx in self.adapter_layers:
                adapter_outputs[str(idx)] = self.adapters[str(idx)](
                    hidden[:, 1:, :],
                    hidden[:, 0, :],
                )
        return adapter_outputs

    def aggregate_evidence(self, adapter_feats: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        query = adapter_feats[str(self.query_layer)].unsqueeze(1)
        evidence = torch.stack(
            [adapter_feats[str(layer)] for layer in self.evidence_layers],
            dim=1,
        )
        attended, attn_weights = self.cross_attn(
            query=query,
            key=evidence,
            value=evidence,
            need_weights=True,
            average_attn_weights=False,
        )
        final_feat = self.cross_attn_norm(query + self.cross_attn_dropout(attended)).squeeze(1)
        return final_feat, attn_weights

    def classifier(self, final_feat: torch.Tensor) -> torch.Tensor:
        return self.head(final_feat)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        text_tokens = data_dict.get('text_tokens', {})

        loss_dict = {}
        total_align_loss = torch.tensor(0.0, device=label.device)

        for layer in self.adapter_layers:
            key = f'L{layer}'
            lam = self.lambda_weights.get(layer, 0.5)
            tokens_l = text_tokens.get(key)

            if tokens_l is None:
                loss_dict[f'align_L{layer}'] = torch.tensor(0.0, device=label.device)
                continue

            valid_mask = (tokens_l.sum(dim=-1) > 0).to(label.device) & (label == 1)
            if valid_mask.sum() == 0:
                loss_dict[f'align_L{layer}'] = torch.tensor(0.0, device=label.device)
                continue

            t_l = self.encode_text(tokens_l)
            v_proj = pred_dict[f'v_proj_{layer}']
            align_loss = self.align_loss_func(v_proj, t_l, valid_mask)

            loss_dict[f'align_L{layer}'] = align_loss
            total_align_loss = total_align_loss + lam * align_loss

        cls_loss = self.cls_loss_func(pred_dict['cls'], label)
        loss_dict['cls'] = cls_loss
        loss_dict['overall'] = total_align_loss + self.beta * cls_loss
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        adapter_feats = self.features(data_dict)
        final_feat, attn_weights = self.aggregate_evidence(adapter_feats)
        logits = self.classifier(final_feat)
        prob = torch.softmax(logits, dim=1)[:, 1]

        pred_dict = {
            'cls': logits,
            'prob': prob,
            'feat': final_feat,
            'adapter_feats': adapter_feats,
            'cross_attn_weights': attn_weights,
        }

        if not inference:
            for layer in self.adapter_layers:
                source_feat = final_feat if layer == self.query_layer else adapter_feats[str(layer)]
                v_proj = self.v_projections[str(layer)](source_feat)
                pred_dict[f'v_proj_{layer}'] = F.normalize(v_proj, dim=-1)

        return pred_dict
