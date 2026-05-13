#!/usr/bin/env python3
"""
agent_server.py
===============
File-IPC inference loop. Watches the Lua side's `agent_io/snapshot.json`,
runs the policy transformer, writes back `agent_io/action.txt`.

Companion to ``Balatro/agent_bridge.lua``. The Lua bridge writes snapshots
into Balatro's Love save directory (e.g.
``%AppData%/Balatro/agent_io/snapshot.json`` on Windows) using atomic rename;
this server polls that file with a short interval, encodes it, runs the
model, then writes ``action.txt`` (also atomically) with the chosen label.

Snapshot schemas:

- ``"live/2.0.0"`` (current) — canonical granularize-3.0 step shape:
  ``objects`` carry canonical zone names including ``PendingCards``;
  ``pending_cards`` is a top-level convenience copy; action labels in
  ``legal_actions`` are zoned (e.g. ``BuyShopItem_VoucherShopOfferings_0``,
  ``SelectCard_CurrentHand_3``, ``SWAP_0_1``).
- ``"live/1.0.0"`` (legacy) — pre-rewrite shape with ``*Selected`` / ``*All``
  zones, ``current_hand_or_pack`` / ``selected_cards`` top-level fields,
  flat action labels like ``BuyShopItem_0`` / ``SellItem_3``. Accepted via
  ``LiveEncoder._normalize_legacy_snapshot`` so a stale Lua build can't
  brick the running server; a single diagnostic line is logged on the
  first legacy snapshot of a session.

Action file format (single line):

    <request_id>\t<action_label>\n

The Lua bridge ignores stale snapshots (request_id mismatch) so a slow
server tick can never desync the game.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model import ModelConfig, PolicyTransformer
from live.live_encoder import LiveEncoder


def default_io_dir() -> Path:
    """Best-guess location of the Lua bridge's IPC dir on this OS."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "Balatro" / "agent_io"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Balatro" / "agent_io"
    return Path.home() / ".local" / "share" / "love" / "Balatro" / "agent_io"


def atomic_write(path: Path, contents: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(contents, encoding="utf-8")
    # Windows can transiently deny os.replace when Balatro has just opened
    # action.txt for reading/deleting. Retry briefly instead of letting the
    # inference server crash and leaving Lua stuck waiting_for_action.
    last_error: OSError | None = None
    for attempt in range(10):
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                last_error = e
        try:
            os.replace(tmp, path)
            return
        except OSError as e:
            last_error = e
            time.sleep(0.01 * (attempt + 1))
    # Best-effort fallback: the rename never succeeded, but a direct write
    # can still succeed once Balatro releases the old file handle. This is
    # not perfectly atomic, so only use it after rename retries are exhausted.
    try:
        path.write_text(contents, encoding="utf-8")
        tmp.unlink(missing_ok=True)
        return
    except OSError:
        tmp.unlink(missing_ok=True)
        if last_error is not None:
            raise last_error
        raise


class AgentServer:
    def __init__(
        self,
        checkpoint_path: Path,
        io_dir: Path,
        device: torch.device,
        snapshot_name: str = "snapshot.json",
        action_name: str = "action.txt",
        runs_log_name: str = "runs.csv",
        decisions_log_name: str = "decisions.ndjson",
        poll_interval_s: float = 0.02,
        verbose: bool = True,
    ) -> None:
        self.encoder = LiveEncoder()
        self.io_dir = io_dir
        self.snapshot_path = io_dir / snapshot_name
        self.action_path = io_dir / action_name
        self.runs_log_path = io_dir / runs_log_name
        self.decisions_log_path = io_dir / decisions_log_name
        self.poll_interval = poll_interval_s
        self.verbose = verbose
        self.device = device

        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        self._last_request_id: int | None = None
        self._legacy_warned: bool = False

    def _load_model(self, ckpt_path: Path) -> PolicyTransformer:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg_dict: dict[str, Any] = ckpt["model_config"]
        # ModelConfig is a dataclass; pass through field names it knows.
        cfg = ModelConfig(**{k: v for k, v in cfg_dict.items()
                             if k in ModelConfig.__dataclass_fields__})
        model = PolicyTransformer(cfg).to(self.device)
        state = ckpt["model_state_dict"]
        model.load_state_dict(state)
        epoch = ckpt.get("epoch")
        val_top1 = ckpt.get("val_top1")
        print(f"[agent_server] loaded {ckpt_path.name} epoch={epoch} val_top1={val_top1}")
        return model

    def _read_snapshot(self) -> dict[str, Any] | None:
        if not self.snapshot_path.exists():
            return None
        try:
            text = self.snapshot_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        # Eagerly delete so we don't reprocess.
        try:
            self.snapshot_path.unlink()
        except OSError:
            pass
        try:
            snap = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[agent_server] snapshot json error: {e}")
            return None
        # Persist a debug copy keyed by page_name (one file per page) so we can
        # compare live snapshots against training samples offline.
        try:
            page = (snap.get("page_name") or "Unknown").replace("/", "_")
            dbg_dir = self.io_dir / "snapshots_debug"
            dbg_dir.mkdir(exist_ok=True)
            dbg_path = dbg_dir / f"latest_{page}.json"
            dbg_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        except OSError:
            pass
        return snap

    @torch.no_grad()
    def _decide(self, snapshot: dict[str, Any]) -> tuple[str, int, list[tuple[str, float]]]:
        batch, legal_mask = self.encoder.encode(snapshot, device=self.device)
        logits = self.model(batch)            # (1, N_ACTIONS)
        # logits already have -inf on illegal slots from masked_fill in model.forward.
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        # Argmax
        action_id = int(probs.argmax())
        # Top-5 for debug log
        top_k = min(5, int(legal_mask.sum()) or 1)
        idxs = np.argsort(-probs)[:top_k]
        topk = [(self.encoder.label_for_index(int(i)), float(probs[int(i)])) for i in idxs]
        return self.encoder.label_for_index(action_id), action_id, topk

    # Per-page "least destructive" fallback action used when _decide raises
    # (e.g. CUDA OOM, vocab miss, malformed snapshot). The goal is just to
    # advance the game out of whatever page the snapshot is on so the next
    # snapshot has a fresh state for the model to retry from.
    _PAGE_PREFERRED_FALLBACK: dict[str, str] = {
        "Cash_Out": "CashOut",
        "Blind_Select": "SelectBlind",
        "In_Shop": "LeaveShop",
        "In_TarotSpectral_Pack": "SkipPack",
        "In_JokerStandardPlanet_Pack": "SkipPack",
        "In_Blind": "DiscardHand",
    }

    def _fallback_label(self, snapshot: dict[str, Any]) -> str:
        legal = snapshot.get("legal_actions") or []
        legal_set = set(legal)
        page = snapshot.get("page_name") or ""
        preferred = self._PAGE_PREFERRED_FALLBACK.get(page)
        if preferred and preferred in legal_set:
            return preferred
        # Anything legal beats freezing. Prefer fixed labels over zoned
        # picks (which might pick an objectively bad joker buy).
        for label in legal:
            if "_" not in label:
                return label
        if legal:
            return legal[0]
        # Snapshot has *no* legal actions. As an absolute last resort, fall
        # back to a couple of universally-safe fixed labels — at worst they
        # are no-ops on the current page.
        return preferred or "LeaveShop"

    def _log_decision(
        self,
        snapshot: dict[str, Any],
        action_label: str,
        action_id: int,
        topk: list[tuple[str, float]],
        elapsed_ms: float,
    ) -> None:
        record = {
            "ts": time.time(),
            "request_id": snapshot.get("request_id"),
            "page_name": snapshot.get("page_name"),
            "n_legal": len(snapshot.get("legal_actions") or []),
            "action_label": action_label,
            "action_id": action_id,
            "top5": [{"label": l, "p": p} for l, p in topk],
            "elapsed_ms": elapsed_ms,
            "run_id": (snapshot.get("meta") or {}).get("run_id"),
            "ante": (snapshot.get("state") or {}).get("ante"),
            "round": (snapshot.get("state") or {}).get("round"),
        }
        try:
            with self.decisions_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def step_once(self) -> bool:
        """Process at most one snapshot. Returns True if work was done."""
        snapshot = self._read_snapshot()
        if snapshot is None:
            return False

        req_id = snapshot.get("request_id")
        if req_id is not None and req_id == self._last_request_id:
            # Stale duplicate (shouldn't happen with atomic rename, but defensive).
            return False
        self._last_request_id = req_id

        schema = snapshot.get("schema_version")
        if schema != "live/2.0.0" and not self._legacy_warned:
            print(
                f"[agent_server] legacy snapshot schema_version={schema!r} — "
                "translating through the live_encoder compatibility shim. "
                "Update Balatro/agent_bridge.lua to emit live/2.0.0 to remove "
                "this hop."
            )
            self._legacy_warned = True

        t0 = time.perf_counter()
        try:
            label, idx, topk = self._decide(snapshot)
        except Exception as e:
            # Don't leave the Lua side hanging — emit a safe fallback so
            # the game keeps moving. We pick the most "harmless" action for
            # the current page from the snapshot's own legal_actions list,
            # then degrade to whatever the first legal label happens to be.
            label = self._fallback_label(snapshot)
            print(
                f"[agent_server] decide error: {e!r} — "
                f"emitting fallback={label!r} so the bot doesn't freeze"
            )
            idx = -1
            topk = [(label, 1.0)]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        line = f"{req_id if req_id is not None else 0}\t{label}\n"
        atomic_write(self.action_path, line)

        self._log_decision(snapshot, label, idx, topk, elapsed_ms)
        if self.verbose:
            top_str = ", ".join(f"{l}={p:.2f}" for l, p in topk[:3])
            print(
                f"[agent_server] req={req_id} page={snapshot.get('page_name')} "
                f"-> {label} (top5: {top_str})  {elapsed_ms:.1f}ms"
            )
        return True

    def serve_forever(self) -> None:
        self.io_dir.mkdir(parents=True, exist_ok=True)
        print(f"[agent_server] watching {self.snapshot_path}")
        print(f"[agent_server] writing  {self.action_path}")
        print(f"[agent_server] device   {self.device}")
        # Touch the decisions log so it exists immediately for tailing.
        if not self.decisions_log_path.exists():
            self.decisions_log_path.touch()
        while True:
            try:
                did_work = self.step_once()
            except KeyboardInterrupt:
                print("\n[agent_server] interrupted, exiting")
                return
            if not did_work:
                time.sleep(self.poll_interval)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO_ROOT / "artifacts" / "checkpoints" / "best.pt",
        help="Path to the .pt training checkpoint.",
    )
    ap.add_argument(
        "--io-dir",
        type=Path,
        default=default_io_dir(),
        help="Directory shared with the Lua agent_bridge "
             "(default: Balatro Love save dir / agent_io).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu | cuda | auto",
    )
    ap.add_argument("--poll-interval", type=float, default=0.02)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    server = AgentServer(
        checkpoint_path=args.checkpoint,
        io_dir=args.io_dir,
        device=device,
        poll_interval_s=args.poll_interval,
        verbose=not args.quiet,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
