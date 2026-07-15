"""Shared blind-token class groupings and cohort helpers."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import numpy as np

# Boss blinds: 370; 372–393; 395–399.  Big blind 371 and small blind 394 are separate.
# (User text "3712–393" is read as 372–393 so 371 stays big blind.)
BOSS_BLIND_CLASS_IDS: frozenset[int] = frozenset({370}) | frozenset(
    range(372, 394)
) | frozenset(range(395, 400))
SMALL_BLIND_CLASS_IDS: frozenset[int] = frozenset({394})
BIG_BLIND_CLASS_IDS: frozenset[int] = frozenset({371})

ALLOWED_BLIND_CLASS_IDS: frozenset[int] = (
    BOSS_BLIND_CLASS_IDS | SMALL_BLIND_CLASS_IDS | BIG_BLIND_CLASS_IDS
)

BLIND_FAMILY_ORDER: tuple[str, ...] = ("boss", "small", "big")

BLIND_TOKEN_ZONE = "BlindToken"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    rad = z * math.sqrt(max(0.0, p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))) / denom
    lo, hi = center - rad, center + rad
    return max(0.0, lo), min(1.0, hi)


def ieee_rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.linewidth": 0.4,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def blind_family_for_class_id(cid: int) -> str | None:
    if cid in BIG_BLIND_CLASS_IDS:
        return "big"
    if cid in SMALL_BLIND_CLASS_IDS:
        return "small"
    if cid in BOSS_BLIND_CLASS_IDS:
        return "boss"
    return None


def load_fps_map(extracted_root: Path) -> dict[str, float]:
    m: dict[str, float] = {}
    for part in sorted(extracted_root.glob("video_id=*")):
        if not part.is_dir():
            continue
        vid = part.name.split("=", 1)[1]
        en = list(part.glob("*_enriched.json"))
        if not en:
            continue
        try:
            d = json.loads(en[0].read_text(encoding="utf-8"))
            m[vid] = float(d.get("fps") or 30)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            m[vid] = 30.0
    return m


def pick_stake(objects: list | None) -> int | None:
    stakes = [
        o
        for o in (objects or [])
        if o.get("zone") == "CurrentStake" and o.get("class_id") is not None
    ]
    if not stakes:
        return None
    stakes.sort(key=lambda o: (o.get("position_in_zone", 0), o.get("slot_id", 0)))
    return int(stakes[-1]["class_id"])


def pick_blind_token_class(objects: list | None) -> int | None:
    toks = [
        o
        for o in (objects or [])
        if o.get("zone") == BLIND_TOKEN_ZONE and o.get("class_id") is not None
    ]
    if not toks:
        return None
    toks.sort(key=lambda o: (o.get("position_in_zone", 0), o.get("slot_id", 0)))
    return int(toks[-1]["class_id"])


def _get_ante_round(ev: dict) -> tuple[int | None, int | None]:
    s = ev.get("state") or {}
    a, r = s.get("ante"), s.get("round")
    try:
        ai = int(a) if a is not None else None
    except (TypeError, ValueError):
        ai = None
    try:
        ri = int(r) if r is not None else None
    except (TypeError, ValueError):
        ri = None
    return ai, ri


def run_has_clean_start(events: list[dict]) -> bool:
    for ev in events:
        a, r = _get_ante_round(ev)
        if a is None or r is None:
            continue
        return a == 1 and r == 0
    return False


def first_stake_class(events: list[dict]) -> int | None:
    for ev in events:
        c = pick_stake(ev.get("objects"))
        if c is not None:
            return c
    return None


def run_max_round(events: list[dict]) -> int | None:
    m: int | None = None
    for ev in events:
        _, r = _get_ante_round(ev)
        if r is None:
            continue
        m = r if m is None else max(m, r)
    return m


def run_final_round(events: list[dict]) -> int | None:
    for ev in reversed(events):
        _, r = _get_ante_round(ev)
        if r is not None:
            return r
    return None


def get_round(ev: dict) -> int | None:
    _, r = _get_ante_round(ev)
    return r


def run_has_round_with_blind_family(
    evs: list[dict], r: int, family: str
) -> bool:
    for ev in evs:
        if get_round(ev) != r:
            continue
        cid = pick_blind_token_class(ev.get("objects"))
        if cid is None:
            continue
        fam = blind_family_for_class_id(cid)
        if fam == family:
            return True
    return False


def last_blind_family_at_round(evs: list[dict], r: int) -> str | None:
    """Chronologically last `BlindToken` family seen while OCR round == r."""
    last: str | None = None
    for ev in evs:
        if get_round(ev) != r:
            continue
        cid = pick_blind_token_class(ev.get("objects"))
        if cid is None:
            continue
        fam = blind_family_for_class_id(cid)
        if fam is not None:
            last = fam
    return last


def collect_loss_tables(
    parsed_root: Path,
    stake_id: int,
    round_max_exclusive: int,
) -> tuple[dict[tuple[int, str], int], dict[tuple[int, str], int], int]:
    """Returns (reached, lost_at) counts keyed by (round, family), n_kept_runs."""
    reached: dict[tuple[int, str], int] = defaultdict(int)
    lost_at: dict[tuple[int, str], int] = defaultdict(int)
    n_kept = 0
    rounds = range(1, round_max_exclusive)
    fams = BLIND_FAMILY_ORDER

    for part in sorted(parsed_root.glob("video_id=*")):
        for run_path in sorted(part.glob("run_*.json")):
            data = json.loads(run_path.read_text(encoding="utf-8"))
            evs = data.get("events") or []
            if not evs or not run_has_clean_start(evs):
                continue
            if first_stake_class(evs) != stake_id:
                continue
            mx = run_max_round(evs)
            fin = run_final_round(evs)
            if mx is None or fin is None:
                continue
            n_kept += 1
            for r in rounds:
                for fam in fams:
                    if mx >= r and run_has_round_with_blind_family(evs, r, fam):
                        reached[(r, fam)] += 1
            lf = last_blind_family_at_round(evs, fin)
            if lf is not None:
                lost_at[(fin, lf)] += 1

    return reached, lost_at, n_kept


def collect_mean_dt_by_round_family(
    parsed_root: Path,
    extracted_root: Path,
    afk_s: float,
    stake_id: int,
    round_max_exclusive: int,
) -> dict[tuple[int, str], list[float]]:
    """Δt samples keyed by (curr_round, blind_family) on current event."""
    fps_map = load_fps_map(extracted_root)
    by_rf: dict[tuple[int, str], list[float]] = defaultdict(list)

    for part in sorted(parsed_root.glob("video_id=*")):
        vid = part.name.split("=", 1)[1]
        fps = fps_map.get(vid) or 30.0
        for run_path in sorted(part.glob("run_*.json")):
            data = json.loads(run_path.read_text(encoding="utf-8"))
            evs = data.get("events") or []
            if not evs or not run_has_clean_start(evs):
                continue
            if first_stake_class(evs) != stake_id:
                continue
            prev_frame: int | None = None
            prev_round: int | None = None
            for ev in evs:
                frame = ev.get("frame_idx")
                if frame is None:
                    continue
                frame = int(frame)
                curr_round = get_round(ev)
                stake = pick_stake(ev.get("objects"))
                blind_cid = pick_blind_token_class(ev.get("objects"))
                fam = (
                    blind_family_for_class_id(blind_cid)
                    if blind_cid is not None
                    else None
                )
                if (
                    prev_frame is not None
                    and curr_round is not None
                    and prev_round is not None
                    and fam is not None
                    and 1 <= curr_round < round_max_exclusive
                    and 1 <= prev_round < round_max_exclusive
                    and stake == stake_id
                ):
                    dt = (frame - prev_frame) / fps
                    if 0.0 <= dt <= afk_s:
                        by_rf[(curr_round, fam)].append(dt)
                prev_frame = frame
                prev_round = curr_round if curr_round is not None else prev_round

    return by_rf


def _rank_average(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return v
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def pearson_spearman(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    try:
        from scipy import stats as st

        pr, pp = st.pearsonr(x, y)
        sr, sp = st.spearmanr(x, y)
        return float(pr), float(pp), float(sr), float(sp)
    except Exception:
        pr_mat = np.corrcoef(x, y)
        pr = float(pr_mat[0, 1]) if pr_mat.shape == (2, 2) else float("nan")
        rx = _rank_average(x)
        ry = _rank_average(y)
        sr_mat = np.corrcoef(rx, ry)
        sr = float(sr_mat[0, 1]) if sr_mat.shape == (2, 2) else float("nan")
        return pr, float("nan"), sr, float("nan")
