import torch
import torch.nn as nn

class KairoTransducer(nn.Module):
    """
    ライブ変換のための Transformer-Transducer (RNN-T) モデル骨組み
    """
    def __init__(
        self,
        input_vocab_size: int,   # 例: アルファベット(a-z) + 記号
        output_vocab_size: int,  # 例: 日本語文字(ひらがな、漢字、記号) + Blank(空白)
        embed_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        # 1. Encoder (Acoustic Modelに相当)
        # ユーザーが打ったローマ字を処理する
        self.encoder_emb = nn.Embedding(input_vocab_size, embed_dim)
        # 本格実装時はTransformerEncoder等に変更
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # 2. Prediction Network (Language Modelに相当)
        # これまでに出力した日本語(確定済み文字)を処理する
        self.pred_emb = nn.Embedding(output_vocab_size, embed_dim)
        # 本格実装時はTransformerDecoderやLSTMに変更
        self.pred_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        
        # 3. Joint Network
        # Encoderの特徴量とPrediction Networkの特徴量を結合して最終予測
        self.joint_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.joint_fc2 = nn.Linear(hidden_dim, output_vocab_size)
        
    def forward(self, x, y):
        """
        x: 入力ローマ字のテンソル (Batch, Time_x)
        y: 出力済み日本語のテンソル (Batch, Time_y)
        """
        # --- Encoder ---
        x_emb = self.encoder_emb(x)
        enc_out, _ = self.encoder_lstm(x_emb) # (Batch, Time_x, Hidden)
        
        # --- Prediction Network ---
        y_emb = self.pred_emb(y)
        pred_out, _ = self.pred_lstm(y_emb)   # (Batch, Time_y, Hidden)
        
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
