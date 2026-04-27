"""
Zoom Decoder model wrapper.

Adds two novelties on top of a standard causal LM:

1. **Aperture Token**: a learnable embedding indexed by size_bucket
   (0..len(SIZE_BUCKETS)-1).  The aperture embedding is prepended to the
   decoder's input embeddings at position 0 and gives the model an explicit
   "focal-length" signal.

2. **Layer truncation**: `num_hidden_layers` can be reduced at init time to
   produce decoders of different sizes from the same pretrained backbone.

The model preserves standard CausalLM behaviour (logits over vocab, same
tokenizer as the base VLM), so it can be dropped into the MemoryDecoder
inference wrapper without changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from zoom_decoder.data_utils import SIZE_BUCKETS


class ZoomDecoder(nn.Module):
    """Causal LM decoder with an optional aperture-size conditioning token."""

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        num_size_buckets: int = len(SIZE_BUCKETS),
        use_aperture: bool = True,
    ):
        super().__init__()
        self.base = base_model
        self.hidden_size = hidden_size
        self.use_aperture = use_aperture
        if use_aperture:
            self.aperture_embed = nn.Embedding(num_size_buckets, hidden_size)
            nn.init.normal_(self.aperture_embed.weight, std=0.02)

    @property
    def config(self):
        return self.base.config

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_layers: Optional[int] = None,
        use_aperture: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "ZoomDecoder":
        """Load and optionally truncate layers to produce a smaller variant."""
        cfg = AutoConfig.from_pretrained(model_name_or_path)
        full_layers = cfg.num_hidden_layers
        if num_layers is not None and num_layers < full_layers:
            cfg.num_hidden_layers = num_layers
            # Some newer configs (e.g. Qwen3) carry a per-layer `layer_types`
            # list that must also be trimmed.
            if hasattr(cfg, "layer_types") and isinstance(getattr(cfg, "layer_types"), list):
                cfg.layer_types = cfg.layer_types[:num_layers]
            base = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, config=cfg, dtype=dtype,
                ignore_mismatched_sizes=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype)
        hidden_size = cfg.hidden_size
        return cls(base, hidden_size=hidden_size, use_aperture=use_aperture)

    def save_pretrained(self, save_dir: str):
        self.base.save_pretrained(save_dir)
        if self.use_aperture:
            torch.save(self.aperture_embed.state_dict(), f"{save_dir}/aperture_embed.pt")

    @classmethod
    def load_from_dir(cls, save_dir: str, dtype: torch.dtype = torch.bfloat16) -> "ZoomDecoder":
        base = AutoModelForCausalLM.from_pretrained(save_dir, dtype=dtype)
        hidden = base.config.hidden_size
        use_ap = False
        try:
            state = torch.load(f"{save_dir}/aperture_embed.pt", map_location="cpu")
            use_ap = True
        except FileNotFoundError:
            state = None
        m = cls(base, hidden_size=hidden, use_aperture=use_ap)
        if state is not None:
            m.aperture_embed.load_state_dict(state)
        return m

    # -----------------------------------------------------------------

    def _embed_with_aperture(
        self, input_ids: torch.Tensor, size_bucket: Optional[torch.Tensor]
    ):
        """Return (inputs_embeds, attention_mask_offset, label_offset).

        If aperture enabled, prepends 1 extra token at position 0.
        """
        tok_embed = self.base.get_input_embeddings()
        embeds = tok_embed(input_ids)  # (B, T, H)
        if not self.use_aperture or size_bucket is None:
            return embeds, 0
        ap = self.aperture_embed(size_bucket).unsqueeze(1)  # (B, 1, H)
        ap = ap.to(embeds.dtype)
        embeds = torch.cat([ap, embeds], dim=1)
        return embeds, 1

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        size_bucket: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        embeds, offset = self._embed_with_aperture(input_ids, size_bucket)

        if attention_mask is not None and offset:
            pad = torch.ones(
                (attention_mask.size(0), offset),
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            attention_mask = torch.cat([pad, attention_mask], dim=1)

        out = self.base(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits  # (B, T+offset, V)
        # Drop the aperture-position logit so downstream code sees (B, T, V).
        if offset:
            logits = logits[:, offset:, :]
        return CausalLMOutputWithPast(logits=logits)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
