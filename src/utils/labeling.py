from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

FOUNDATIONS = ["CH", "FC", "LB", "AS", "PD"]
NON_MORAL = "NM"
LABELS_6 = FOUNDATIONS + [NON_MORAL]

LABEL2ID = {l: i for i, l in enumerate(LABELS_6)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def extract_label_from_row(row: Dict) -> Tuple[int, str]:
    """Map dataset row with *_ref 0/1 labels into a single 6-way label.

    We restrict to 'clean' rows:
    - either exactly one of CH/FC/LB/AS/PD is 1
    - OR none of them are 1 and non_moral_ref==1 (NM)

    Returns:
        (label_id, label_name)
    Raises:
        ValueError if row is not clean.
    """
    refs = [int(row.get(f"{k}_ref", 0)) for k in FOUNDATIONS]
    non_moral = int(row.get("non_moral_ref", 0))
    s = sum(refs)

    if s == 0:
        if non_moral != 1:
            raise ValueError("Not clean: no foundation but non_moral_ref!=1")
        return LABEL2ID[NON_MORAL], NON_MORAL

    if s == 1 and non_moral == 0:
        idx = refs.index(1)
        name = FOUNDATIONS[idx]
        return LABEL2ID[name], name

    raise ValueError("Not clean: multi-label or conflicting non_moral")
