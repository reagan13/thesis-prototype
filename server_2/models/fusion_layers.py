import torch
import torch.nn as nn

# Concatenation Fusion Layer
class FusionLayer(nn.Module):
    def __init__(self, gpt2_dim: int, bert_dim: int, output_dim: int, dropout_rate: float):
        super().__init__()
        self.gpt2_proj = nn.Linear(gpt2_dim, output_dim)
        self.bert_proj = nn.Linear(bert_dim, output_dim)
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, gpt2_features: torch.Tensor, bert_features: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        gpt2_proj = self.gpt2_proj(gpt2_features)
        bert_proj = self.bert_proj(bert_features)
        concat_features = torch.cat([gpt2_proj, bert_proj], dim=-1)
        fused = self.fusion(concat_features)
        return self.layer_norm(fused)

    pass


# Cross-Attention Fusion Layer
class CrossAttentionFusionLayer(nn.Module):
    def __init__(self, gpt2_dim: int, bert_dim: int, output_dim: int, dropout_rate: float, num_heads: int = 8):
        super().__init__()
        self.gpt2_proj = nn.Linear(gpt2_dim, output_dim)
        self.bert_proj = nn.Linear(bert_dim, output_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, gpt2_features: torch.Tensor, bert_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        gpt2_proj = self.gpt2_proj(gpt2_features)
        bert_proj = self.bert_proj(bert_features)
        attn_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, 0)
        fused_features, _ = self.cross_attention(
            query=gpt2_proj,
            key=bert_proj,
            value=bert_proj,
            key_padding_mask=attention_mask == 0
        )
        fused_features = self.dropout(fused_features) + gpt2_proj
        return self.layer_norm(fused_features)
    pass

# Dense Fusion Layer
class DenseFusionLayer(nn.Module):
    def __init__(self, gpt2_dim: int, bert_dim: int, output_dim: int, dropout_rate: float):
        super().__init__()
        self.gpt2_proj = nn.Linear(gpt2_dim, output_dim)
        self.bert_proj = nn.Linear(bert_dim, output_dim)
        self.dense = nn.Linear(output_dim, output_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, gpt2_features: torch.Tensor, bert_features: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        gpt2_proj = self.gpt2_proj(gpt2_features)
        bert_proj = self.bert_proj(bert_features)
        combined_features = gpt2_proj + bert_proj
        fused_features = self.dense(combined_features)
        fused_features = self.activation(fused_features)
        fused_features = self.dropout(fused_features)
        return self.layer_norm(fused_features)
    pass