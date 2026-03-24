from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class EncoderBundle:
    model_id: str
    tokenizer: any
    model: any
    device: torch.device


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_encoder(model_id: str, device: torch.device) -> EncoderBundle:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return EncoderBundle(model_id=model_id, tokenizer=tok, model=model, device=device)


@dataclass
class TranslatorBundle:
    model_id: str
    tokenizer: any
    model: any
    device: torch.device


def load_translator(model_id: str, device: torch.device) -> TranslatorBundle:
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return TranslatorBundle(model_id=model_id, tokenizer=tok, model=model, device=device)


@torch.no_grad()
def translate_batch(
    bundle: TranslatorBundle,
    texts: List[str],
    max_length: int = 256,
    batch_size: int = 16,
) -> List[str]:
    """Translate a list of texts using a Marian/NLLB-like seq2seq model."""
    outs: List[str] = []
    tok = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=max_length)
        dec = tok.batch_decode(gen, skip_special_tokens=True)
        outs.extend(dec)
    return outs
