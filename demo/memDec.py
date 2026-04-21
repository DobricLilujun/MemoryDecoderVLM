from typing import Optional, Tuple, List, Union, FrozenSet
import torch
import torch.nn.functional as F
from torch import nn
from loguru import logger
from dataclasses import dataclass

from transformers import (
    GenerationMixin,
    PreTrainedModel,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    GenerationConfig,
)
from transformers.utils import ModelOutput

@dataclass
class MemoryDecoderOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    knn_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class MemoryDecoder(PreTrainedModel, GenerationMixin):
    """
    A light wrapper around a base model (LLM **or** VLM) and a text‑only
    memory decoder that fuses their logits:

        logits_joint = logaddexp(logits_base + log(1‑λ),
                                 logits_mem  + log(λ))

    For VLMs the visual inputs (pixel_values, image_grid_thw, …) are
    forwarded **only** to the base model.  The memory decoder always
    receives text‑only inputs.

    Greedy decoding chooses argmax over `logits_joint`.
    """

    # Keys that carry visual information → passed to base_lm only.
    VISUAL_KEYS: FrozenSet[str] = frozenset({
        'pixel_values', 'pixel_values_videos',
        'image_grid_thw', 'video_grid_thw',
        'image_sizes', 'images', 'image_embeds',
        'image_token_indices', 'image_bound',
        'vision_feature_layer', 'vision_feature_select_strategy',
        'tgt_sizes', 'image_flags',
        'rope_deltas',
    })

    def __init__(
        self,
        base_lm,
        knn_generator,
        lmbda: float = 0.25,
        knn_temp: float = 1.0,
    ):
        config = base_lm.config
        # Avoid SDPA check for this wrapper (no real attention layers)
        config._attn_implementation = "eager"
        super().__init__(config)

        self.base_lm = base_lm
        self.knn_generator = knn_generator
        self.lmbda = float(lmbda)
        self.knn_temp = float(knn_temp)

    # ------------------------------------------------------------------ #
    #  helper: split kwargs into visual / text‑only
    # ------------------------------------------------------------------ #
    def _split_kwargs(self, kwargs):
        text_kwargs = {k: v for k, v in kwargs.items()
                       if k not in self.VISUAL_KEYS}
        return text_kwargs          # base_lm receives the full kwargs

    # ------------------------------------------------------------------ #
    #  helper: align vocab sizes when base & mem decoder differ
    # ------------------------------------------------------------------ #
    @staticmethod
    def _align_vocab(logits_a, logits_b):
        va, vb = logits_a.shape[-1], logits_b.shape[-1]
        if va != vb:
            v = min(va, vb)
            logits_a = logits_a[..., :v]
            logits_b = logits_b[..., :v]
        return logits_a, logits_b

    # ------------------------------------------------------------------ #
    #                       1. forward()
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        knn_past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        Forward pass that returns **fused log‑probs** as logits.
        We keep separate caches for each sub‑model.
        Visual kwargs are forwarded only to the base model.
        """
        text_kwargs = self._split_kwargs(kwargs)

        # --- base model (LLM or VLM) receives ALL kwargs ----------------
        base_outputs = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        # --- memory decoder receives text‑only kwargs -------------------
        knn_outputs = self.knn_generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=knn_past_key_values,
            use_cache=use_cache,
            **text_kwargs,
        )

        # Temperature on memory decoder logits only
        logits_base = base_outputs.logits      # (B, T, V_base)
        logits_knn  = knn_outputs.logits       # (B, T, V_mem)
        if self.knn_temp != 1.0:
            logits_knn = logits_knn / self.knn_temp

        # Align vocab sizes (VLM may have extra vision tokens)
        logits_base, logits_knn = self._align_vocab(logits_base, logits_knn)

        # Convert to log‑probabilities first (numerically stable when fusing)
        logp_base = F.log_softmax(logits_base, dim=-1)
        logp_knn  = F.log_softmax(logits_knn, dim=-1)

        logp_joint = torch.logaddexp(
            logp_base + torch.log(torch.tensor(1.0 - self.lmbda, device=logp_base.device)),
            logp_knn  + torch.log(torch.tensor(self.lmbda, device=logp_base.device)),
        )

        return MemoryDecoderOutput(
            logits=logp_joint,
            past_key_values=base_outputs.past_key_values,
            knn_past_key_values=knn_outputs.past_key_values,
            hidden_states=None,
            attentions=None
        )
        
    # ------------------------------------------------------------------ #
    #                       2. generate()
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 20,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        do_sample: bool = False,             # must be False (greedy) for now
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        """
        Greedy decoding with **shared** stopping criteria.
        We keep two independent KV caches (one per sub‑model) and extend them
        step‑by‑step.
        For VLMs, visual inputs are consumed only in the first forward pass;
        subsequent steps rely on the KV cache.
        """
        if do_sample:
            raise ValueError("MemoryDecoder.generate only supports greedy decoding (do_sample=False).")

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # ---- first forward (includes visual inputs if any) ------------- #
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        next_token_logits = outputs["logits"][:, -1, :]            # (B, V)

        base_past = outputs["past_key_values"]
        knn_past = outputs["knn_past_key_values"]

        # Greedy select
        next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # (B,1)
        generated = torch.cat([input_ids, next_tokens], dim=-1)              # (B,T+1)

        # Remove visual kwargs – they are now encoded in the KV cache.
        subsequent_kwargs = {k: v for k, v in kwargs.items()
                            if k not in self.VISUAL_KEYS}

        # --- main loop -------------------------------------------------- #
        num_new_token = 0
        while True:
            if stopping_criteria is not None and False not in stopping_criteria(generated, None):
                break
            if num_new_token >= max_new_tokens:
                break

            outputs = self.forward(
                input_ids=next_tokens,
                attention_mask=None,          # past manages causal masking
                past_key_values=base_past,
                knn_past_key_values=knn_past,
                use_cache=True,
                **subsequent_kwargs,
            )
            next_token_logits = outputs["logits"][:, -1, :]
            base_past = outputs["past_key_values"]
            knn_past = outputs["knn_past_key_values"]

            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_tokens], dim=-1)
            num_new_token += 1

        return generated