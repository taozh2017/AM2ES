import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.model_utils import *

class FeatureMaskExpert(nn.Module):
    """单个专家：对 input_dim 每个维度产生一个 mask logit"""
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or max(128, input_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) 
        )

    def forward(self, x):
        return self.net(x)


class FeatureLevelMoE(nn.Module):
    """Feature-level MoE 融合模块"""
    def __init__(self, input_dim: int = 256*3, out_dim: int = 256, num_experts: int = 4,
                 expert_hidden: Optional[int] = None, gate_hidden: Optional[int] = None,
                 temperature: float = 1.0, expert_dropout: float = 0.0, top_k: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.expert_dropout = expert_dropout
        self.top_k = top_k 

        self.experts = nn.ModuleList([
            FeatureMaskExpert(input_dim, hidden_dim=expert_hidden) for _ in range(num_experts)
        ])
        
        gate_hidden = gate_hidden or max(64, input_dim // 8)
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, num_experts)
        )

        self.reducer = nn.Linear(input_dim, out_dim)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(float(temperature))), requires_grad=False)

    def forward(self, x: torch.Tensor, return_aux: bool = True):
        K, device, E = x.size(0), x.device, self.num_experts
        temp = torch.exp(self.log_temp)

        # 1) Experts masks
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        expert_masks = torch.sigmoid(expert_logits / (temp + 1e-12)) 

        # 2) Gating weights
        gate_logits = self.gate_net(x) 
        gate_w = F.softmax(gate_logits / (temp + 1e-12), dim=-1)

        # Expert Dropout (Training)
        if self.training and self.expert_dropout > 0:
            drop_mask = (torch.rand(E, device=device) >= self.expert_dropout).float()
            gate_w = gate_w * drop_mask.unsqueeze(0)
            gate_w = gate_w / gate_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # 3) Top-K Hard selection (Eval)
        if not self.training and self.top_k is not None and self.top_k < E:
            topk_vals, topk_idx = torch.topk(gate_w, self.top_k, dim=-1)
            new_gate = torch.zeros_like(gate_w)
            new_gate.scatter_(1, topk_idx, topk_vals)
            gate_w = new_gate / (topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-6))

        # 4) Combine masks & apply
        gate_w_exp = gate_w.unsqueeze(-1)
        combined_mask = (gate_w_exp * expert_masks).sum(dim=1) 
        out = self.reducer(x * combined_mask) 

        if return_aux:
            mask_sparsity = combined_mask.mean()
            gate_entropy = -(gate_w * (gate_w + 1e-12).log()).sum(dim=-1).mean()
            return out, dict(mask=combined_mask, gate_w=gate_w, mask_sparsity=mask_sparsity, gate_entropy=gate_entropy)
        return out

class TokenFusionMoE(nn.Module):
    """Token-level MoE 融合"""
    def __init__(self, input_dim=256, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, input_dim * 2), nn.ReLU(), nn.Linear(input_dim * 2, input_dim))
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_scores = torch.softmax(self.gate(x), dim=-1)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=2)
        weighted_expert_out = (expert_outs * gate_scores.unsqueeze(1)).sum(dim=2)
        return weighted_expert_out.mean(dim=0), {'gate_scores': gate_scores.detach(), 'expert_outs': expert_outs.detach()}


class ExpertScorer(nn.Module):
    """基础专家评分模块，为每个特征向量打分 (Aligned with Sparse Top-k Routing)"""
    def __init__(self, input_dim=256, num_experts=4, top_k_experts=2):
        super().__init__()
        self.top_k_experts = top_k_experts
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.ReLU(), nn.Linear(input_dim // 2, 1))
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        if self.top_k_experts < self.num_experts:
            topk_vals, topk_idx = torch.topk(gate_logits, self.top_k_experts, dim=-1)
            sparse_logits = torch.full_like(gate_logits, -1e9)
            sparse_logits.scatter_(-1, topk_idx, topk_vals)
        else:
            sparse_logits = gate_logits
            
        gate_score = torch.softmax(sparse_logits, dim=-1)
        expert_outputs = torch.cat([expert(x) for expert in self.experts], dim=1)
        return torch.sum(gate_score * expert_outputs, dim=1)


def select_top_features(x, scorer, top_k=4):
    scores = scorer(x)
    topk_idx = torch.topk(scores, k=top_k, dim=0).indices
    return x[topk_idx], scores[topk_idx]


class TransformerCluster(nn.Module):
    def __init__(self, input_dim=256, n_clusters=64, num_heads=8, dropout=0.1):
        super().__init__()
        self.cluster_tokens = nn.Parameter(torch.randn(1, n_clusters, input_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(input_dim * 4, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(0) 
            
        B = x.size(0)
        q = self.cluster_tokens.expand(B, -1, -1)
        
        attn_out, _ = self.cross_attn(query=q, key=x, value=x)
        out = self.norm1(q + self.dropout1(attn_out))
        out_clusters = self.norm2(out + self.dropout2(self.ffn(out)))
        
        if is_2d:
            out_clusters = out_clusters.squeeze(0)
        return out_clusters


class Anchor_Bridge(nn.Module):
    def __init__(self, embed_dim=256, n_anchor=6, n_cluster=64, n_heads=4, n_layers=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_anchor = n_anchor

        self.rna_proj = nn.Linear(embed_dim, embed_dim)
        self.img_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj_rna = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj_img = nn.Linear(embed_dim, embed_dim, bias=False)

        self.global_anchors = nn.Parameter(torch.randn(n_anchor, embed_dim) * 0.01)
        self.delta_gen = nn.Sequential(
            nn.Linear(embed_dim * 2, max(256, embed_dim * 2)), nn.ReLU(), nn.Linear(max(256, embed_dim * 2), n_anchor * embed_dim)
        )

        self.cluster_pool = TransformerCluster(embed_dim, n_cluster, n_heads, n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.act = nn.ReLU()

    def make_anchors(self, x_rna, x_img_cluster):
        rna_pool = x_rna.mean(dim=0)
        if x_img_cluster is not None and x_img_cluster.numel() > 0:
            cond = torch.cat([rna_pool, x_img_cluster.mean(dim=0)], dim=-1)
        else:
            cond = rna_pool

        delta = self.delta_gen(cond).view(self.n_anchor, self.embed_dim)
        return self.global_anchors + delta

    def forward(self, x_rna, x_img):
        x_img_cluster = self.cluster_pool(x_img) 

        k_rna, k_img = self.rna_proj(x_rna), self.img_proj(x_img_cluster)
        anchors = self.make_anchors(x_rna, x_img_cluster)

        q_anchor_rna, q_anchor_img = self.q_proj_rna(anchors), self.q_proj_img(anchors)

        attn_rna = F.softmax((q_anchor_rna @ k_rna.T) / (self.embed_dim ** 0.5), dim=-1)
        attn_img = F.softmax((q_anchor_img @ k_img.T) / (self.embed_dim ** 0.5), dim=-1)

        fused = self.act(self.fusion(torch.cat([attn_rna @ k_rna, attn_img @ k_img], dim=-1)))
        return self.norm(fused + anchors), attn_rna, attn_img, anchors

    def anchor_regularization(self, anchors):
        norm_anchors = F.normalize(anchors, dim=-1)
        sim_matrix = torch.matmul(norm_anchors, norm_anchors.T)
        identity = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        
        orth_loss = ((sim_matrix - identity) ** 2).mean()
        sparse_loss = anchors.abs().mean()
        return orth_loss, sparse_loss


class FixedAnchorBridge(nn.Module):
    """AD-MAES w/ Fixed Anchors"""
    def __init__(self, embed_dim=256, n_anchor=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_anchor = n_anchor
        self.global_anchors = nn.Parameter(torch.randn(n_anchor, embed_dim) * 0.01)
        
        self.rna_proj = nn.Linear(embed_dim, embed_dim)
        self.img_proj = nn.Linear(embed_dim, embed_dim)
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU()

    def forward(self, x_rna, x_img):
        anchors = self.global_anchors
        q, k = self.rna_proj(x_rna), self.img_proj(x_img)

        attn_rna = F.softmax(anchors @ q.T / (self.embed_dim ** 0.5), dim=-1)
        attn_img = F.softmax(anchors @ k.T / (self.embed_dim ** 0.5), dim=-1)

        fused = self.act(self.fusion(torch.cat([attn_rna @ q, attn_img @ k], dim=-1)))
        return self.norm(fused + anchors), attn_rna, attn_img, anchors

    def anchor_regularization(self, anchors):
        norm_anchors = F.normalize(anchors, dim=-1)
        sim_matrix = torch.matmul(norm_anchors, norm_anchors.T)
        identity = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        return ((sim_matrix - identity) ** 2).mean(), anchors.abs().mean()


class AM2ES(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(AM2ES, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes

        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        size = self.size_dict_WSI[model_size_wsi]
        hidden = self.size_dict_omic[model_size_omic]

        ### WSI Feature Extractor
        self.wsi_net = nn.Sequential(
            nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25)
        )
        
        ### Genomic SNN Feature Extractors
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### Dynamic Anchor Bridges
        self.tsb_5x = Anchor_Bridge(embed_dim=256, n_anchor=16, n_cluster=64)
        self.tsb_10x = Anchor_Bridge(embed_dim=256, n_anchor=16, n_cluster=128)
        self.tsb_20x = Anchor_Bridge(embed_dim=256, n_anchor=16, n_cluster=256)

        ### Feature Selectors & MoE
        self.moe_scorer = ExpertScorer(input_dim=256, num_experts=4, top_k_experts=2)
        self.token_fusion_moe = TokenFusionMoE(input_dim=256, num_experts=4)
        self.feature_moe = FeatureLevelMoE(
            input_dim=256*3, out_dim=256, num_experts=4,
            expert_hidden=384, gate_hidden=128,
            temperature=0.5, expert_dropout=0.2
        )

        self.path_rho = nn.Sequential(nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout))
        
        self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)

        self.identity = nn.Identity()
        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        # 提取输入
        x_path_5x, x_path_10x, x_path_20x = kwargs['x_path_5x'], kwargs['x_path_10x'], kwargs['x_path_20x']
        x_omic = [kwargs[f'x_omic{i}'] for i in range(1, 7)]
        
        # WSIs降维处理
        h_bag_5x = self.wsi_net(x_path_5x)
        h_bag_10x = self.wsi_net(x_path_10x)
        h_bag_20x = self.wsi_net(x_path_20x)
        
        # 基因网络处理与堆叠
        h_omic_bag = torch.stack([net(feat) for net, feat in zip(self.sig_networks, x_omic)])
        
        # 跨模态 Anchor Bridge 融合与正则化
        tsb_fused_5x, attn_rna_5x, attn_img_5x, anchors_5x = self.tsb_5x(h_omic_bag, h_bag_5x)
        orth_loss_5x, sparse_loss_5x = self.tsb_5x.anchor_regularization(anchors_5x)

        tsb_fused_10x, attn_rna_10x, attn_img_10x, anchors_10x = self.tsb_10x(h_omic_bag, h_bag_10x)
        orth_loss_10x, sparse_loss_10x = self.tsb_10x.anchor_regularization(anchors_10x)

        tsb_fused_20x, attn_rna_20x, attn_img_20x, anchors_20x = self.tsb_20x(h_omic_bag, h_bag_20x)
        orth_loss_20x, sparse_loss_20x = self.tsb_20x.anchor_regularization(anchors_20x)

        # 阶段一：尺度内专家选择
        sel_5x, score_5x = select_top_features(tsb_fused_5x, self.moe_scorer, top_k=4)
        sel_10x, score_10x = select_top_features(tsb_fused_10x, self.moe_scorer, top_k=4)
        sel_20x, score_20x = select_top_features(tsb_fused_20x, self.moe_scorer, top_k=4)

        # 阶段二：跨尺度融合消融控制
        x_fused = torch.cat([sel_5x, sel_10x, sel_20x], dim=1) 
        final_feature = self.feature_moe(x_fused, return_aux=False) 

        # 下游分类器
        h = self.path_rho(final_feature.mean(dim=0))
        logits = self.classifier(h).unsqueeze(0)
        
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        aux = {
            'orth_loss_5x': orth_loss_5x, 'sparse_loss_5x': sparse_loss_5x,
            'orth_loss_10x': orth_loss_10x, 'sparse_loss_10x': sparse_loss_10x,
            'orth_loss_20x': orth_loss_20x, 'sparse_loss_20x': sparse_loss_20x
        }
        
        return hazards, S, Y_hat, aux