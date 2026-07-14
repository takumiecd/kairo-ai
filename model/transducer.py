import torch
import torch.nn as nn

from model.profile_encoder import DEFAULT_TOP_K
from model.profile_encoder import ProfileEncoder
from model.transformer_encoder import TransformerSequenceEncoder

class KairoTransducer(nn.Module):
    """
    ライブ変換のための Transformer-Transducer (RNN-T) モデル骨組み

    Encoder / Prediction Network はそれぞれ ``"lstm"`` / ``"transformer"`` を
    選べる。Joint Network と損失・デコード経路は共通なので、中身だけ差し替えて
    A/B 比較できる。設計方針は ``docs/MODEL_DESIGN.md`` を参照。

    ``profile_conditioning=True`` (docs/PROFILE.md §4, 段階B) にすると、
    プロファイル埋め込み e(u) を Prediction Network の入力先頭に
    プレフィックストークンとして注入する。デフォルトは ``False`` で、
    既存の重み名・forward の挙動を完全に維持する(``profile_encoder`` 等の
    追加パラメータも一切登録されない)。
    """
    def __init__(
        self,
        input_vocab_size: int,   # 例: アルファベット(a-z) + 記号
        output_vocab_size: int,  # 例: 日本語文字(ひらがな、漢字、記号) + Blank(空白)
        embed_dim: int | None = 256,
        hidden_dim: int | None = 512,
        input_embed_dim: int | None = None,
        output_embed_dim: int | None = None,
        encoder_hidden_dim: int | None = None,
        prediction_hidden_dim: int | None = None,
        joint_hidden_dim: int | None = None,
        encoder_type: str = "lstm",
        prediction_type: str = "lstm",
        encoder_layers: int = 2,
        prediction_layers: int = 1,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.1,
        max_positions: int = 256,
        input_pad_id: int = 0,
        output_pad_id: int = 0,
        profile_conditioning: bool = False,
        profile_top_k: int = DEFAULT_TOP_K,
    ):
        super().__init__()
        input_embed_dim = input_embed_dim or int(embed_dim)
        output_embed_dim = output_embed_dim or int(embed_dim)
        encoder_hidden_dim = encoder_hidden_dim or int(hidden_dim)
        prediction_hidden_dim = prediction_hidden_dim or int(hidden_dim)
        joint_hidden_dim = joint_hidden_dim or int(hidden_dim)
        self.encoder_type = encoder_type
        self.prediction_type = prediction_type
        self.profile_conditioning = profile_conditioning

        # 1. Encoder (Acoustic Modelに相当)
        # ユーザーが打ったローマ字を処理する。入力は確定済みなので双方向(未来を見てよい)。
        if encoder_type == "lstm":
            self.encoder_emb = nn.Embedding(input_vocab_size, input_embed_dim)
            self.encoder_lstm = nn.LSTM(input_embed_dim, encoder_hidden_dim, num_layers=encoder_layers, batch_first=True)
        elif encoder_type == "transformer":
            self.encoder_transformer = TransformerSequenceEncoder(
                vocab_size=input_vocab_size,
                embed_dim=input_embed_dim,
                model_dim=encoder_hidden_dim,
                pad_id=input_pad_id,
                num_layers=encoder_layers,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                max_positions=max_positions,
                causal=False,
            )
        else:
            raise ValueError(f"unknown encoder_type: {encoder_type!r}")

        # 2. Prediction Network (Language Modelに相当)
        # これまでに出力した日本語(確定済み文字)を処理する。逐次生成なので causal。
        if prediction_type == "lstm":
            self.pred_emb = nn.Embedding(output_vocab_size, output_embed_dim)
            self.pred_lstm = nn.LSTM(output_embed_dim, prediction_hidden_dim, num_layers=prediction_layers, batch_first=True)
        elif prediction_type == "transformer":
            self.pred_transformer = TransformerSequenceEncoder(
                vocab_size=output_vocab_size,
                embed_dim=output_embed_dim,
                model_dim=prediction_hidden_dim,
                pad_id=output_pad_id,
                num_layers=prediction_layers,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                max_positions=max_positions,
                causal=True,
            )
        else:
            raise ValueError(f"unknown prediction_type: {prediction_type!r}")

        # 2.5 プロファイルエンコーダ (docs/PROFILE.md §4, 段階B)
        # profile_conditioning=False のときは一切構築しない
        # (既存チェックポイントとの state_dict 互換を壊さないため)。
        if self.profile_conditioning:
            self.profile_encoder = ProfileEncoder(
                embed_dim=output_embed_dim,
                output_dim=prediction_hidden_dim,
                top_k=profile_top_k,
            )
            if self.prediction_type == "lstm":
                # LSTM の入力は output_embed_dim 次元。e(u) は
                # prediction_hidden_dim 次元なので、通常の埋め込みトークンと
                # 同じ入力空間へ落としてから先頭に連結する。
                self.profile_lstm_proj = nn.Linear(prediction_hidden_dim, output_embed_dim)

        # 3. Joint Network
        # Encoderの特徴量とPrediction Networkの特徴量を結合して最終予測
        self.joint_fc1 = nn.Linear(encoder_hidden_dim + prediction_hidden_dim, joint_hidden_dim)
        self.joint_fc2 = nn.Linear(joint_hidden_dim, output_vocab_size)

    def encode(self, x):
        """x: (B, T_x) → エンコーダ特徴 (B, T_x, encoder_hidden_dim)。"""
        if self.encoder_type == "lstm":
            enc_out, _ = self.encoder_lstm(self.encoder_emb(x))
            return enc_out
        return self.encoder_transformer(x)

    def predict(self, y, profile_features=None):
        """y: (B, T_y) → Prediction 特徴 (B, T_y, prediction_hidden_dim)。

        ``profile_features``: e(u) 本体、shape (B, prediction_hidden_dim)。
        ``None``(デフォルト)のときは既存の計算と完全に同一(後方互換)。
        """
        if self.prediction_type == "lstm":
            embedded = self.pred_emb(y)  # (B, T_y, output_embed_dim)
            if profile_features is not None:
                prefix = self.profile_lstm_proj(profile_features).unsqueeze(1)  # (B, 1, E)
                embedded = torch.cat([prefix, embedded], dim=1)
            pred_out, _ = self.pred_lstm(embedded)
            if profile_features is not None:
                pred_out = pred_out[:, 1:, :]
            return pred_out
        return self.pred_transformer(y, prefix_embed=profile_features)

    def forward(self, x, y, profile_features=None):
        """
        x: 入力ローマ字のテンソル (Batch, Time_x)
        y: 出力済み日本語のテンソル (Batch, Time_y)
        profile_features: 段階B条件付け用の入力一式(dict)、もしくは ``None``。
            ``{"domain", "lang", "word_char_ids", "word_char_mask", "word_mask"}``
            を持つ(``model.profile_encoder.encode_profile_batch`` が組み立てる
            形式)。``profile_conditioning=False`` のときは無視される。
        """
        # --- Encoder ---
        enc_out = self.encode(x)  # (Batch, Time_x, Hidden)

        # --- プロファイルエンコーダ (段階B, profile_conditioning=True のみ) ---
        e_u = None
        if self.profile_conditioning and profile_features is not None:
            embedding = self.pred_emb if self.prediction_type == "lstm" else self.pred_transformer.emb
            e_u = self.profile_encoder(
                domain=profile_features["domain"],
                lang=profile_features["lang"],
                word_char_ids=profile_features["word_char_ids"],
                word_char_mask=profile_features["word_char_mask"],
                word_mask=profile_features["word_mask"],
                embedding=embedding,
            )

        # --- Prediction Network ---
        pred_out = self.predict(y, profile_features=e_u)  # (Batch, Time_y, Hidden)

        # --- Joint Network ---
        # 計算を効率化するため、ブロードキャストで結合 (Batch, Time_x, Time_y, Hidden*2)
        enc_out_exp = enc_out.unsqueeze(2)    # (B, T_x, 1, H)
        pred_out_exp = pred_out.unsqueeze(1)  # (B, 1, T_y, H)
        
        # タイミングの全組み合わせを結合 (学習時のみ必要。推論時は1ステップずつ行う)
        enc_out_exp = enc_out_exp.expand(-1, -1, pred_out.size(1), -1)
        pred_out_exp = pred_out_exp.expand(-1, enc_out.size(1), -1, -1)
        joint_in = torch.cat([enc_out_exp, pred_out_exp], dim=-1)
        
        # 最終確率分布へ
        out = torch.tanh(self.joint_fc1(joint_in))
        logits = self.joint_fc2(out) # (B, T_x, T_y, Vocab)
        
        return logits

# Note:
# 実際の学習では torchaudio.functional.rnnt_loss を使用してLossを計算します。
# 推論時は Beam Search を用いてこのネットワークを1ステップずつ動かします。
