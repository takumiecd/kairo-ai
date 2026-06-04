"""Context-aware discrete denoising model for IME conversion."""

from __future__ import annotations

import torch
import torch.nn as nn


class KairoDiffusionModel(nn.Module):
    """Recover a clean output canvas from a noisy one at any diffusion timestep."""

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        model_dim: int = 256,
        input_embed_dim: int = 64,
        output_embed_dim: int = 256,
        num_heads: int = 4,
        num_input_layers: int = 3,
        num_context_layers: int = 2,
        num_canvas_layers: int = 3,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_positions: int = 256,
        diffusion_steps: int = 8,
    ) -> None:
        super().__init__()
        self.max_positions = max_positions
        self.diffusion_steps = diffusion_steps

        self.input_emb = nn.Embedding(input_vocab_size, input_embed_dim)
        self.input_proj = nn.Linear(input_embed_dim, model_dim)
        self.output_emb = nn.Embedding(output_vocab_size, output_embed_dim)
        self.output_proj = nn.Linear(output_embed_dim, model_dim)
        self.input_pos = nn.Embedding(max_positions, model_dim)
        self.output_pos = nn.Embedding(max_positions, model_dim)
        self.time_emb = nn.Embedding(diffusion_steps + 1, model_dim)

        input_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        context_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        canvas_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.input_encoder = nn.TransformerEncoder(input_layer, num_input_layers)
        self.context_encoder = nn.TransformerEncoder(context_layer, num_context_layers)
        self.canvas_decoder = nn.TransformerDecoder(canvas_layer, num_canvas_layers)

        self.token_head = nn.Linear(model_dim, output_vocab_size)
        self.length_head = nn.Linear(model_dim, max_positions + 1)
        self.no_context = nn.Parameter(torch.zeros(1, 1, model_dim))

    def _positions(self, length: int, device: torch.device) -> torch.Tensor:
        if length > self.max_positions:
            raise ValueError(
                f"sequence length {length} exceeds max_positions={self.max_positions}"
            )
        return torch.arange(length, device=device)

    def encode_condition(
        self,
        inputs: torch.Tensor,
        contexts: torch.Tensor,
        input_pad_mask: torch.Tensor | None = None,
        context_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_pos = self._positions(inputs.shape[1], inputs.device)
        input_hidden = self.input_proj(self.input_emb(inputs)) + self.input_pos(input_pos)[None]
        input_memory = self.input_encoder(input_hidden, src_key_padding_mask=input_pad_mask)

        if contexts.shape[1] == 0:
            context_memory = self.no_context.expand(inputs.shape[0], -1, -1)
            context_pad_mask = torch.zeros(
                (inputs.shape[0], 1), dtype=torch.bool, device=inputs.device
            )
        else:
            context_pos = self._positions(contexts.shape[1], contexts.device)
            context_hidden = (
                self.output_proj(self.output_emb(contexts))
                + self.output_pos(context_pos)[None]
            )
            if context_pad_mask is None:
                context_pad_mask = torch.zeros(
                    contexts.shape, dtype=torch.bool, device=contexts.device
                )
            context_memory = self.context_encoder(
                context_hidden, src_key_padding_mask=context_pad_mask
            )

        memory = torch.cat([context_memory, input_memory], dim=1)
        if input_pad_mask is None:
            input_pad_mask = torch.zeros(
                inputs.shape, dtype=torch.bool, device=inputs.device
            )
        memory_pad_mask = torch.cat([context_pad_mask, input_pad_mask], dim=1)
        pooled = self._masked_mean(memory, memory_pad_mask)
        return memory, memory_pad_mask, pooled

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        keep = (~pad_mask).unsqueeze(-1).to(hidden.dtype)
        return (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        inputs: torch.Tensor,
        contexts: torch.Tensor,
        noisy_canvas: torch.Tensor,
        timesteps: torch.Tensor,
        input_pad_mask: torch.Tensor | None = None,
        context_pad_mask: torch.Tensor | None = None,
        canvas_pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory, memory_pad_mask, pooled = self.encode_condition(
            inputs, contexts, input_pad_mask, context_pad_mask
        )
        canvas_pos = self._positions(noisy_canvas.shape[1], noisy_canvas.device)
        canvas_hidden = (
            self.output_proj(self.output_emb(noisy_canvas))
            + self.output_pos(canvas_pos)[None]
            + self.time_emb(timesteps)[:, None]
        )
        decoded = self.canvas_decoder(
            canvas_hidden,
            memory,
            tgt_key_padding_mask=canvas_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        return {
            "token_logits": self.token_head(decoded),
            "length_logits": self.length_head(pooled),
        }
