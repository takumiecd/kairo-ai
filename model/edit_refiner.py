"""Neural edit refiner: 反復・非自己回帰で IME 仮説を書き換えるモデル。

設計方針（議論のまとめ）:
- 左→右の自己回帰や「カーソルを単調に進める編集スクリプト」はやめる。
  代わりに **直前の変換仮説をシード** にして、全位置を **並列** に書き換える
  (Levenshtein Transformer 流)。レイテンシは文長ではなく **ラウンド数** で
  決まるため、リアルタイム IME に向く。
- ローマ字側 (input) と 仮説側 (hypothesis) を別エンコーダで埋め込み、
  仮説エンコーダから **クロスアテンション** でローマ字を参照する。
  「いまどの文字を直すか」は 1 本の文脈ベクトルではなく **位置ごとの表現**
  h_i に根ざして判断する（旧 edit_transducer の汎化失敗を回避）。

1 ラウンドは LevT と同じ 3 つの並列ヘッド:
  1. delete : 既存トークンごとに KEEP / DELETE
  2. insert : 隣接トークン間の各ギャップに「何個挿入するか」(0..K)
  3. fill   : 挿入したプレースホルダに実トークンを予測（穴埋め）

delete+insert で系列の「形」を決め、プレースホルダを挿んだ系列を
**再エンコード** して fill する（LevT と同じ 2 パス構成）。
推論は仮説をシードに 1〜2 ラウンドだけ回し、上限ラウンドでハード停止する。
"""

from __future__ import annotations

import torch
import torch.nn as nn


# delete ヘッドのクラス
KEEP = 0
DELETE = 1


class KairoEditRefiner(nn.Module):
    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        placeholder_id: int,
        model_dim: int = 256,
        input_embed_dim: int = 64,
        output_embed_dim: int = 256,
        num_heads: int = 4,
        num_input_layers: int = 3,
        num_hypothesis_layers: int = 3,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_insertions_per_gap: int = 8,
        max_positions: int = 256,
    ) -> None:
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.placeholder_id = placeholder_id
        self.max_insertions_per_gap = max_insertions_per_gap

        # --- 埋め込み（入出力で別次元）+ 学習位置埋め込み ---
        # 入力(ローマ字)は a-z+記号で語彙が小さいので低次元、出力(tokenizer 後の
        # 日本語; char/BPE)は語彙が大きいので高次元、と別々に持つ。
        # cross-attention と self-attn のために、各々を共通の model_dim へ射影する。
        self.input_emb = nn.Embedding(input_vocab_size, input_embed_dim)
        self.input_proj = nn.Linear(input_embed_dim, model_dim)
        # 仮説側の埋め込みはプレースホルダ <plh> を含むため output 語彙を使う。
        self.hypothesis_emb = nn.Embedding(output_vocab_size, output_embed_dim)
        self.hypothesis_proj = nn.Linear(output_embed_dim, model_dim)
        # 位置埋め込みは射影後の model_dim 上で足す。
        self.input_pos = nn.Embedding(max_positions, model_dim)
        self.hypothesis_pos = nn.Embedding(max_positions, model_dim)

        # --- ローマ字エンコーダ（自己アテンションのみ） ---
        # ストリーミングしたい場合は causal マスク + KV キャッシュに差し替える。
        input_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.input_encoder = nn.TransformerEncoder(input_layer, num_input_layers)

        # --- 仮説エンコーダ（双方向 self-attn + ローマ字への cross-attn） ---
        # TransformerDecoderLayer に causal マスクを渡さない = 双方向 self-attn。
        hypothesis_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.hypothesis_encoder = nn.TransformerDecoder(hypothesis_layer, num_hypothesis_layers)

        # --- 3 つの編集ヘッド（model_dim 上で動く） ---
        self.delete_head = nn.Linear(model_dim, 2)
        # 挿入数はギャップ表現（隣接トークンの結合）から 0..K を分類。
        self.insert_head = nn.Linear(model_dim * 2, max_insertions_per_gap + 1)
        self.fill_head = nn.Linear(model_dim, output_vocab_size)

    # ------------------------------------------------------------------
    # エンコード
    # ------------------------------------------------------------------
    def encode_input(
        self,
        romaji: torch.Tensor,
        romaji_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """romaji: (B, S) -> memory: (B, S, model_dim)"""
        positions = torch.arange(romaji.shape[1], device=romaji.device)
        emb = self.input_proj(self.input_emb(romaji)) + self.input_pos(positions)[None]
        return self.input_encoder(emb, src_key_padding_mask=romaji_pad_mask)

    def encode_hypothesis(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        tokens_pad_mask: torch.Tensor | None = None,
        memory_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """tokens: (B, T) -> 位置ごと表現 h: (B, T, model_dim)"""
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        emb = self.hypothesis_proj(self.hypothesis_emb(tokens)) + self.hypothesis_pos(positions)[None]
        return self.hypothesis_encoder(
            emb,
            memory,
            tgt_key_padding_mask=tokens_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )

    # ------------------------------------------------------------------
    # ヘッド
    # ------------------------------------------------------------------
    def delete_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D) -> (B, T, 2)  各既存トークンの KEEP/DELETE。"""
        return self.delete_head(hidden)

    def insert_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D) -> (B, T-1, K+1)  隣接トークン間ギャップの挿入数。

        仮説は BOS/EOS で挟む前提なので、T-1 個のギャップで全ての実ギャップ
        （先頭・末尾を含む）を覆う。
        """
        gap = torch.cat([hidden[:, :-1], hidden[:, 1:]], dim=-1)  # (B, T-1, 2D)
        return self.insert_head(gap)

    def fill_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """h: (B, T', D) -> (B, T', V)  プレースホルダ位置の穴埋め。"""
        return self.fill_head(hidden)

    # ------------------------------------------------------------------
    # forward（学習用）: 2 パスをまとめて返す
    # ------------------------------------------------------------------
    def forward(
        self,
        romaji: torch.Tensor,
        hypothesis: torch.Tensor,
        placeholder_tokens: torch.Tensor | None = None,
        romaji_pad_mask: torch.Tensor | None = None,
        hypothesis_pad_mask: torch.Tensor | None = None,
        placeholder_pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """学習時の forward。

        - delete / insert は元の hypothesis から予測。
        - fill は「プレースホルダを挿入済みの系列」(placeholder_tokens) を
          再エンコードして予測する（LevT の 2 パス目）。教師は最小編集
          オラクルから作る想定（train 側で用意）。
        """
        memory = self.encode_input(romaji, romaji_pad_mask)
        hidden = self.encode_hypothesis(
            hypothesis, memory, hypothesis_pad_mask, romaji_pad_mask
        )
        outputs = {
            "delete_logits": self.delete_logits(hidden),
            "insert_logits": self.insert_logits(hidden),
        }
        if placeholder_tokens is not None:
            filled_hidden = self.encode_hypothesis(
                placeholder_tokens, memory, placeholder_pad_mask, romaji_pad_mask
            )
            outputs["fill_logits"] = self.fill_logits(filled_hidden)
        return outputs

    # ------------------------------------------------------------------
    # 推論: 1 系列を 1 ラウンド精製（バッチ版も同形で書ける）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def refine_once(
        self,
        romaji_ids: list[int],
        hypothesis_ids: list[int],
    ) -> tuple[list[int], bool]:
        """hypothesis を 1 ラウンド書き換える。

        hypothesis_ids は BOS/EOS で挟まれている前提（両端は編集しない）。
        戻り値: (新しい hypothesis_ids, changed)
        changed=False なら収束（呼び出し側は停止してよい）。
        """
        device = next(self.parameters()).device
        romaji = torch.tensor([romaji_ids], dtype=torch.long, device=device)
        memory = self.encode_input(romaji)

        tokens = torch.tensor([hypothesis_ids], dtype=torch.long, device=device)
        hidden = self.encode_hypothesis(tokens, memory)

        # --- 1) 削除 ---
        delete = self.delete_logits(hidden)[0].argmax(dim=-1)  # (T,)
        delete[0] = KEEP   # BOS は保持
        delete[-1] = KEEP  # EOS は保持
        kept = [tid for tid, op in zip(hypothesis_ids, delete.tolist()) if op == KEEP]

        # --- 2) 挿入数（削除後の形に対して測り直す） ---
        tokens = torch.tensor([kept], dtype=torch.long, device=device)
        hidden = self.encode_hypothesis(tokens, memory)
        insert = self.insert_logits(hidden)[0].argmax(dim=-1).tolist()  # (len(kept)-1,)

        with_placeholders: list[int] = [kept[0]]
        for gap_index, count in enumerate(insert):
            with_placeholders.extend([self.placeholder_id] * count)
            with_placeholders.append(kept[gap_index + 1])

        changed = (len(kept) != len(hypothesis_ids)) or any(c > 0 for c in insert)
        if self.placeholder_id not in with_placeholders:
            return with_placeholders, changed

        # --- 3) 穴埋め（プレースホルダ入り系列を再エンコード） ---
        seq = torch.tensor([with_placeholders], dtype=torch.long, device=device)
        filled_hidden = self.encode_hypothesis(seq, memory)
        fill = self.fill_logits(filled_hidden)[0].argmax(dim=-1).tolist()  # (T', V)->(T',)
        result = [
            fill[i] if tid == self.placeholder_id else tid
            for i, tid in enumerate(with_placeholders)
        ]
        return result, changed

    @torch.no_grad()
    def refine(
        self,
        romaji_ids: list[int],
        hypothesis_ids: list[int],
        max_rounds: int = 2,
    ) -> list[int]:
        """ウォームスタート反復精製。max_rounds でハード停止（レイテンシ保証）。"""
        current = hypothesis_ids
        for _ in range(max_rounds):
            current, changed = self.refine_once(romaji_ids, current)
            if not changed:
                break
        return current


# Note:
# - 学習は最小編集オラクル (train/edit_data.py の build_min_edit_script) を
#   位置整列のラベルに変換して、delete / insert-count / fill を同時に教える
#   (imitation learning)。余力があれば LevT 流の roll-in で自分のミスにも強くする。
# - ONNX/Rust へ出す時は、可変ラウンド・可変挿入数が動的グラフになって扱いづらい。
#   max_insertions_per_gap と max_rounds を固定上限にして静的グラフ化するのが実務的。
