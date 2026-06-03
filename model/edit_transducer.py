"""Neural edit transducer model."""

from __future__ import annotations

import torch
import torch.nn as nn

from train.edit_data import ACTION_BOS


class KairoEditTransducer(nn.Module):
    """Predict cursor-based edit actions for revising an IME hypothesis."""

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        input_embed_dim: int = 64,
        output_embed_dim: int = 64,
        action_embed_dim: int = 32,
        hidden_dim: int = 128,
        num_ops: int = 4,
    ) -> None:
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.num_ops = num_ops

        self.input_emb = nn.Embedding(input_vocab_size, input_embed_dim)
        self.previous_emb = nn.Embedding(output_vocab_size, output_embed_dim)
        self.action_op_emb = nn.Embedding(ACTION_BOS + 1, action_embed_dim)
        self.action_insert_emb = nn.Embedding(output_vocab_size, output_embed_dim)

        self.input_encoder = nn.LSTM(input_embed_dim, hidden_dim, batch_first=True)
        self.previous_encoder = nn.LSTM(output_embed_dim, hidden_dim, batch_first=True)
        self.context_to_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.context_to_cell = nn.Linear(hidden_dim * 2, hidden_dim)

        self.action_decoder = nn.LSTM(
            action_embed_dim + output_embed_dim,
            hidden_dim,
            batch_first=True,
        )
        self.op_head = nn.Linear(hidden_dim, num_ops)
        self.insert_head = nn.Linear(hidden_dim, output_vocab_size)

    def _encode_context(self, inputs: torch.Tensor, previous_tokens: torch.Tensor):
        input_emb = self.input_emb(inputs)
        _input_out, (input_hidden, _input_cell) = self.input_encoder(input_emb)
        if previous_tokens.shape[1] == 0:
            previous_hidden = input_hidden.new_zeros(input_hidden.shape)
        else:
            previous_emb = self.previous_emb(previous_tokens)
            _previous_out, (previous_hidden, _previous_cell) = self.previous_encoder(previous_emb)
        context = torch.cat([input_hidden[-1], previous_hidden[-1]], dim=-1)
        hidden = torch.tanh(self.context_to_hidden(context)).unsqueeze(0)
        cell = torch.tanh(self.context_to_cell(context)).unsqueeze(0)
        return hidden, cell

    def forward(
        self,
        inputs: torch.Tensor,
        previous_tokens: torch.Tensor,
        action_input_ops: torch.Tensor,
        action_input_insert_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = self._encode_context(inputs, previous_tokens)
        action_emb = torch.cat(
            [
                self.action_op_emb(action_input_ops),
                self.action_insert_emb(action_input_insert_tokens),
            ],
            dim=-1,
        )
        decoded, _state = self.action_decoder(action_emb, (hidden, cell))
        return self.op_head(decoded), self.insert_head(decoded)

    @torch.no_grad()
    def predict_next(
        self,
        inputs: torch.Tensor,
        previous_tokens: torch.Tensor,
        action_context_ops: torch.Tensor,
        action_context_insert_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        op_logits, insert_logits = self(
            inputs,
            previous_tokens,
            action_context_ops,
            action_context_insert_tokens,
        )
        return op_logits[:, -1], insert_logits[:, -1]
