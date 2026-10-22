"""
LPAA - Layer-wise Progressive Alignment Adapter Detector
放置位置：training/detectors/lpaa_detector.py
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
#  横向适配器：GAP + 两层MLP + 残差，论文公式(1)
# ════════════════════════════════════════════════════════

class HorizontalAdapter(nn.Module):
    def __init__(self, hidden_dim=768, bottleneck=64, dropout=0.1):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states, cls_token):
        # hidden_states:(B,N,D)  cls_token:(B,D)
        x   = self.gap(hidden_states.transpose(1, 2)).squeeze(-1)  # (B,D)
        v_l = cls_token + self.mlp(self.norm(x))                   # (B,D)
        return v_l


# ════════════════════════════════════════════════════════
#  层级对比损失，论文公式(2)
# ════════════════════════════════════════════════════════

class LayerAlignLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, v_l, t_l, valid_mask):
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=v_l.device, requires_grad=True)
        v      = F.normalize(v_l[valid_mask], dim=-1)
        t      = F.normalize(t_l[valid_mask], dim=-1)
        sim    = torch.matmul(v, t.T) / self.tau
        labels = torch.arange(sim.shape[0], device=v.device)
        return F.cross_entropy(sim, labels)


# ════════════════════════════════════════════════════════
#  LPAA Detector 主体
# ════════════════════════════════════════════════════════

@DETECTOR.register_module(module_name='lpaa')
class LPAADetector(AbstractDetector):
    HIDDEN_DIM = 768   # CLIP ViT-B/16

    def __init__(self, config):
        super().__init__()
        self.config         = config
        self.adapter_layers = config.get('adapter_layers', [3, 6, 9, 12])
        self.temperature    = config.get('temperature', 0.07)
        self.beta           = config.get('beta', 1.0)
        self.lambda_weights = config.get('lambda_weights',
                                         {3: 0.5, 6: 0.5, 9: 0.5, 12: 0.5})
        bottleneck = config.get('adapter_bottleneck', 64)
        dropout    = config.get('dropout', 0.1)
        clip_name  = config.get('clip_model_name', 'openai/clip-vit-base-patch16')

        # ── 冻结 CLIP 视觉主干 ──
        self.backbone = self.build_backbone(config)

        # ── 四个横向适配器 ──
        self.adapters = nn.ModuleDict({
            str(l): HorizontalAdapter(self.HIDDEN_DIM, bottleneck, dropout)
            for l in self.adapter_layers
        })

        # ── 视觉→文本空间投影（每层一个，可训练）──
        self.v_projections = nn.ModuleDict({
            str(l): nn.Linear(self.HIDDEN_DIM, 512)
            for l in self.adapter_layers
        })

        # ── 分类头（基于 L12 特征）──
        self.head = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

        # ── 损失函数 ──
        self.cls_loss_func   = self.build_loss(config)
        self.align_loss_func = LayerAlignLoss(self.temperature)

        # ── 冻结 CLIP 文本编码器 ──
        clip_model = CLIPModel.from_pretrained(clip_name)
        self.text_encoder    = clip_model.text_model
        self.text_projection = clip_model.text_projection
        for p in self.text_encoder.parameters():    p.requires_grad = False
        for p in self.text_projection.parameters(): p.requires_grad = False

        logger.info(f"[LPAA] 层={self.adapter_layers} λ={self.lambda_weights} β={self.beta}")

    # ── build_backbone ──
    def build_backbone(self, config):
        clip_name = config.get('clip_model_name', 'openai/clip-vit-base-patch16')
        vision    = CLIPModel.from_pretrained(clip_name).vision_model
        for p in vision.parameters():
            p.requires_grad = False
        logger.info("[LPAA] CLIP 视觉主干已冻结")
        return vision

    # ── build_loss ──
    def build_loss(self, config):
        return LOSSFUNC[config.get('loss_func', 'cross_entropy')]()

    # ── 文本编码（冻结，不参与梯度）──
    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 确保文本编码器和输入在同一设备
        device = next(self.head.parameters()).device
        self.text_encoder    = self.text_encoder.to(device)
        self.text_projection = self.text_projection.to(device)
        input_ids = input_ids.to(device)
        out       = self.text_encoder(input_ids=input_ids)
        projected = self.text_projection(out.pooler_output)
        return F.normalize(projected, dim=-1)

    # ── 视觉特征提取（逐层抽取适配器输出）──
    def features(self, data_dict: dict) -> dict:
        hidden = self.backbone.embeddings(data_dict['image'])
        hidden = self.backbone.pre_layrnorm(hidden)

        # CLIP ViT encoder layer 需要 attention_mask 和 causal_attention_mask
        B, N, _ = hidden.shape
        attention_mask       = None
        causal_attention_mask = None

        adapter_outputs = {}
        for idx, layer in enumerate(self.backbone.encoder.layers, start=1):
            layer_out = layer(
                hidden,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
            )
            hidden = layer_out[0]                              # (B,N+1,D)
            if idx in self.adapter_layers:
                v_l = self.adapters[str(idx)](
                    hidden[:, 1:, :],   # patch tokens
                    hidden[:, 0, :]     # CLS token
                )
                adapter_outputs[str(idx)] = v_l               # (B,D)
        return adapter_outputs

    # ── 分类（用 L12 适配器输出）──
    def classifier(self, features: dict) -> torch.Tensor:
        return self.head(features['12'])

    # ── 损失计算 ──
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label         = data_dict['label']
        adapter_feats = pred_dict['adapter_feats']
        text_tokens   = data_dict.get('text_tokens', {})

        loss_dict        = {}
        total_align_loss = torch.tensor(0.0, device=label.device)

        for l in self.adapter_layers:
            key      = f'L{l}'
            lam      = self.lambda_weights.get(l, 0.5)
            tokens_l = text_tokens.get(key)   # (B,77) or None

            if tokens_l is None:
                loss_dict[f'align_L{l}'] = torch.tensor(0.0, device=label.device)
                continue

            # 有效样本：label=1 且描述非空（非全零 token）
            valid_mask = (tokens_l.sum(dim=-1) > 0).to(label.device) & (label == 1)
            if valid_mask.sum() == 0:
                loss_dict[f'align_L{l}'] = torch.tensor(0.0, device=label.device)
                continue

            t_l        = self.encode_text(tokens_l)           # (B,512)
            v_proj     = pred_dict[f'v_proj_{l}']             # (B,512)
            align_loss = self.align_loss_func(v_proj, t_l, valid_mask)

            loss_dict[f'align_L{l}'] = align_loss
            total_align_loss         = total_align_loss + lam * align_loss

        cls_loss             = self.cls_loss_func(pred_dict['cls'], label)
        loss_dict['cls']     = cls_loss
        loss_dict['overall'] = total_align_loss + self.beta * cls_loss
        return loss_dict

    # ── 训练指标 ──
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred  = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    # ── 前向传播 ──
    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        adapter_feats = self.features(data_dict)
        logits        = self.classifier(adapter_feats)
        prob          = torch.softmax(logits, dim=1)[:, 1]

        pred_dict = {
            'cls':           logits,
            'prob':          prob,
            'feat':          adapter_feats['12'],
            'adapter_feats': adapter_feats,
        }

        # 训练时计算各层视觉投影（推理时不需要）
        if not inference:
            for l in self.adapter_layers:
                v_proj = self.v_projections[str(l)](adapter_feats[str(l)])
                pred_dict[f'v_proj_{l}'] = F.normalize(v_proj, dim=-1)

        return pred_dict