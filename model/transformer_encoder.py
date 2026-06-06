"""差し替え可能な Transformer 系列エンコーダ。

RNN-T の Encoder（ローマ字入力／双方向）と Prediction Network
（出力済み日本語／causal）の両方を、この1モジュールで賄う。違いは
``causal`` フラグだけ:

- Encoder: 入力は打ち終わっているので未来を見てよい → ``causal=False``。
- Prediction: 逐次生成なので未来を見せない → ``causal=True``。

設計の背景は ``docs/MODEL_DESIGN.md`` を参照。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """上三角が ``True`` の (L, L) bool マスク。各位置が自分より後ろを見るのを禁じる。

    padding mask（bool）と型を揃えるため bool で返す（``True`` = 注意禁止）。
    """
    mask = torch.ones((length, length), dtype=torch.bool, device=device)
    return torch.triu(mask, diagonal=1)


class TransformerSequenceEncoder(nn.Module):
    """埋め込み + 学習位置埋め込み + TransformerEncoder の薄いスタック。

    ``model_dim`` で出力するので、LSTM 版の hidden_dim と同じ値を渡せば
    Joint Network 側を一切変更せずに差し替えられる。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        model_dim: int,
        pad_id: int,
        num_layers: int,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        max_positions: int = 256,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.causal = causal
        self.max_positions = max_positions

        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, model_dim) if embed_dim != model_dim else nn.Identity()
        self.pos = nn.Embedding(max_positions, model_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or model_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) のトークンID → (B, T, model_dim)。"""
        length = tokens.shape[1]
        if length > self.max_positions:
            raise ValueError(
                f"sequence length {length} exceeds max_positions={self.max_positions}"
            )

        positions = torch.arange(length, device=tokens.device)
        hidden = self.proj(self.emb(tokens)) + self.pos(positions)[None]

        # <pad> に注意を奪われないよう無視させる。
        pad_mask = tokens == self.pad_id
        attn_mask = build_causal_mask(length, tokens.device) if self.causal else None

        return self.net(hidden, mask=attn_mask, src_key_padding_mask=pad_mask)
