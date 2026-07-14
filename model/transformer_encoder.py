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

    def forward(
        self,
        tokens: torch.Tensor,
        prefix_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """tokens: (B, T) のトークンID → (B, T, model_dim)。

        ``prefix_embed`` (段階B条件付け, docs/PROFILE.md §4): 指定すると
        ``model_dim`` 次元の1ベクトルを系列の先頭に**プレフィックストークン**
        として注入する(位置0)。causal でも全位置から見える(先頭なので未来
        マスクの影響を受けない)。出力は挿入前と同じ ``(B, T, model_dim)`` に
        戻すため、注入した先頭位置は内部で落とす。``None`` のとき(デフォルト)
        は既存の計算経路と完全に同一(後方互換)。
        """
        length = tokens.shape[1]
        if length > self.max_positions:
            raise ValueError(
                f"sequence length {length} exceeds max_positions={self.max_positions}"
            )
        total_length = length + (1 if prefix_embed is not None else 0)
        if total_length > self.max_positions:
            raise ValueError(
                f"sequence length {total_length} (including profile prefix) "
                f"exceeds max_positions={self.max_positions}"
            )

        positions = torch.arange(total_length, device=tokens.device)
        token_positions = positions[1:] if prefix_embed is not None else positions
        token_hidden = self.proj(self.emb(tokens)) + self.pos(token_positions)[None]

        # <pad> に注意を奪われないよう無視させる。
        pad_mask = tokens == self.pad_id

        if prefix_embed is not None:
            batch_size = tokens.shape[0]
            prefix_hidden = (prefix_embed + self.pos(positions[0])[None]).unsqueeze(1)
            hidden = torch.cat([prefix_hidden, token_hidden], dim=1)
            prefix_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device)
            pad_mask = torch.cat([prefix_pad, pad_mask], dim=1)
        else:
            hidden = token_hidden

        attn_mask = build_causal_mask(hidden.shape[1], tokens.device) if self.causal else None
        out = self.net(hidden, mask=attn_mask, src_key_padding_mask=pad_mask)
        if prefix_embed is not None:
            out = out[:, 1:, :]
        return out
