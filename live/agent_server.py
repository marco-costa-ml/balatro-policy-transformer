#!/usr/bin/env python3
"""
agent_server.py
===============
File-IPC inference loop. Watches the Lua side's `agent_io/snapshot.json`,
runs the branched policy transformer, writes back `agent_io/action.txt`.

Companion to ``Balatro/agent_bridge.lua``. The Lua bridge writes snapshots
into Balatro's Love save directory (e.g.
``%AppData%/Balatro/agent_io/snapshot.json`` on Windows) using atomic rename;
this server polls that file with a short interval, encodes it, runs the
model, then writes ``action.txt`` (also atomically) with the chosen label.

Branched policy plan-cache
--------------------------
With the parent-commit branched policy, a single model decision unrolls
into 1..N granular IPC labels (e.g. ``PlayHand`` with 3 cards becomes
``SelectCard_CurrentHand_X`` x3 then ``PlayHand``). Lua still expects
one label per poll, so the server caches the unrolled plan in
``EmissionPolicy`` and emits one label per tick until the plan is
drained; only then does it run the model again.

When the cached plan is invalidated mid-flight (Lua's legal set diverges
or the page transitions unexpectedly) the cache is dropped and the next
poll re-queries the model.

Snapshot schemas:

- ``"live/2.0.0"`` (current) — canonical granularize-3.0 step shape.
- ``"live/1.0.0"`` (legacy) — pre-rewrite shape rerouted via the
  compatibility shim in ``live/live_encoder.py``.

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

from history_features import HistoryBuffer, history_caps
from live.emission_policy import EmissionPolicy, expand_decision
from live.live_encoder import LiveEncoder
from model import ModelConfig, PolicyTransformer


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
        history_steps, _ = history_caps(self.encoder.feature_config)
        self.history_buffer = HistoryBuffer(maxlen=history_steps)
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

        # Plan-cache state machine. ``EmissionPolicy`` holds the queue
        # of granular IPC labels we still owe Lua for the most recent
        # parent-commit decision.
        self.emission = EmissionPolicy()
        self._last_request_id: int | None = None
        self._last_run_id: Any | None = None
        self._legacy_warned: bool = False

    def _load_model(self, ckpt_path: Path) -> PolicyTransformer:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        schema = ckpt.get("schema_version")
        # The branched-policy code path expects a v2.x checkpoint. v1
        # checkpoints carried a single flat ``head`` layer whose weights
        # don't map onto family_head + per-shape decoders, so a silent
        # load would either crash on a state_dict key mismatch or — worse
        # — leave the new heads at random init. Fail fast with a clear
        # message so the user knows to retrain (or invoke the v1 server
        # against the older checkpoint).
        if schema is None or not str(schema).startswith("2."):
            raise RuntimeError(
                f"checkpoint {ckpt_path.as_posix()} has schema_version="
                f"{schema!r}; the live branched-policy server requires "
                "schema 2.x. Retrain with train.py (which writes 2.0.0 "
                "checkpoints) or swap to a pre-branched build of agent_server.py."
            )
        cfg_dict: dict[str, Any] = ckpt["model_config"]
        cfg = ModelConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in ModelConfig.__dataclass_fields__
        })
        model = PolicyTransformer(cfg).to(self.device)
        state = ckpt["model_state_dict"]
        model.load_state_dict(state)
        epoch = ckpt.get("epoch")
        val_top1 = ckpt.get("val_top1")
        family_map_version = ckpt.get("family_map_version")
        print(
            f"[agent_server] loaded {ckpt_path.name} epoch={epoch} "
            f"val_top1={val_top1} family_map={family_map_version} "
            f"schema={schema}"
        )
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

    # ------------------------------------------------------------------
    # Branched decode (two-phase)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decide_super_step(
        self,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the model and return a high-level decision dict.

        Decision dict shape (all int / list[int]):
            {
                "family_id": int,
                "family_name": str,
                "decoder_shape": str,
                "num_cards": int,
                "card_ptr_local_seq": list[int],
                "item_ptr_local": int,
                "swap_i_local": int,
                "swap_j_local": int,
                "family_top5": list[tuple[str, float]],
            }
        """
        batch, _legal_mask = self.encoder.encode(
            snapshot,
            device=self.device,
            history_steps=self.history_buffer.records(),
        )

        # Phase 1: encode + family argmax.
        phase1 = self.model.encode_and_pick_family(batch)
        family_id = int(phase1["family_id"].item())
        family_name = self.encoder.id_to_family[family_id]
        decoder_shape = self.encoder.decoder_shapes.get(family_name, "reserved")

        # Top-5 family for diagnostics.
        family_probs = torch.softmax(phase1["family_logits"], dim=-1).squeeze(0)
        top_n = min(5, int((phase1["family_logits"].squeeze(0) > -float("inf")).sum().item()) or 1)
        top_idx = torch.topk(family_probs, top_n).indices.tolist()
        family_top5: list[tuple[str, float]] = [
            (self.encoder.id_to_family[int(i)], float(family_probs[int(i)].item()))
            for i in top_idx
        ]

        # Phase 2: recompute pointer masks for the predicted family, then
        # run the argument decoders.
        canonical_snap = self.encoder.normalize_snapshot(snapshot)
        lua_mask = self.encoder._legal_action_mask(
            canonical_snap.get("legal_actions") or []
        )
        new_masks = self.encoder.pointer_masks_for_family(
            canonical_snap,
            family_name,
            device=self.device,
            lua_legal_mask=lua_mask if lua_mask.any() else None,
        )
        for key, tensor in new_masks.items():
            batch[key] = tensor

        phase2 = self.model.decode_arguments(phase1["enc"], batch, phase1["family_id"])

        decision: dict[str, Any] = {
            "family_id": family_id,
            "family_name": family_name,
            "decoder_shape": decoder_shape,
            "family_top5": family_top5,
            "item_ptr_local": int(phase2["item_pred"].item()),
            "swap_i_local": int(phase2["swap_i_pred"].item()),
            "swap_j_local": int(phase2["swap_j_pred"].item()),
        }

        # Pick the right card-seq head based on decoder shape.
        if decoder_shape == "card_seq":
            n = int(phase2["card_seq_num_cards"].item())
            n = max(1, n)  # PlayHand/DiscardHand can never be 0-card.
            n = min(n, self.model.cfg.max_cards_per_decision)
            seq = phase2["card_seq_pred"].squeeze(0).tolist()
            decision["num_cards"] = n
            decision["card_ptr_local_seq"] = list(map(int, seq[:n]))
        elif decoder_shape == "chained_cards":
            n = int(phase2["chained_num_cards"].item())
            n = max(0, min(n, self.model.cfg.max_cards_per_decision))
            seq = phase2["chained_pred"].squeeze(0).tolist()
            decision["num_cards"] = n
            decision["card_ptr_local_seq"] = list(map(int, seq[:n]))
        else:
            decision["num_cards"] = 0
            decision["card_ptr_local_seq"] = []

        return decision

    # Per-page "least destructive" fallback action used when the model
    # produces a plan that diverges from Lua's legal set, or when no
    # legal labels match the decoded plan. The goal is just to advance
    # the game out of whatever page the snapshot is on so the next
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

    # ------------------------------------------------------------------
    # History commit (per super-step, not per IPC label)
    # ------------------------------------------------------------------

    def _commit_history_for_plan(self, plan_snapshot: dict[str, Any], decision: dict[str, Any]) -> None:
        """Append a single history record summarising the completed plan.

        The encoder always conditions on PRIOR super-steps' decisions
        (history is built up by parent commit, not by granular IPC), so
        we synthesise one record at the end of each plan from:

        - the snapshot the model conditioned on (``plan_snapshot``),
        - the parent commit label,
        - any selected cards (lifted to ``PendingCards`` so the
          relevant_history_objects helper picks them up).
        """
        family = decision["family_name"]
        shape = decision["decoder_shape"]
        item_ptr = decision.get("item_ptr_local")
        card_seq = list(decision.get("card_ptr_local_seq") or [])

        # Build commit label (the LAST label of the plan, drives history).
        if shape == "no_args":
            commit_label = family
        elif shape == "card_seq":
            commit_label = family
        elif shape == "single_ptr":
            commit_label = f"{family}_{int(item_ptr)}"
        elif shape == "chained_cards":
            commit_label = f"{family}_{int(item_ptr)}"
        elif shape == "joker_pair":
            i, j = int(decision["swap_i_local"]), int(decision["swap_j_local"])
            lo, hi = (i, j) if i < j else (j, i)
            commit_label = f"SWAP_{lo}_{hi}"
        else:
            commit_label = family

        # Lift selected card positions into PendingCards in the record's
        # objects copy so history_features sees "these are the touched
        # cards". We don't reach into the actual encoder zone here; the
        # history builder is robust to slightly-stale object positions.
        step_record = self.encoder.build_step(plan_snapshot)
        step_record = dict(step_record)
        step_record["objects"] = list(step_record.get("objects") or [])
        if card_seq:
            zone_name = "CurrentHand"
            if family == "UseConsumable_CurrentConsumables" and plan_snapshot.get("page_name") == "In_TarotSpectral_Pack":
                zone_name = "TarotSpectralHand"
            elif family == "SelectPackItem_PackOfferings":
                zone_name = "TarotSpectralHand"
            # Find the original objects at those positions in their zone
            # and re-emit them under PendingCards. This is best-effort;
            # mismatches just mean the history sees fewer pending tokens.
            originals_by_pos: dict[int, dict[str, Any]] = {}
            for obj in step_record["objects"]:
                if isinstance(obj, dict) and obj.get("zone") == zone_name:
                    pos = obj.get("position_in_zone")
                    if isinstance(pos, int):
                        originals_by_pos[pos] = obj
            for slot, ptr in enumerate(card_seq):
                src = originals_by_pos.get(int(ptr))
                if isinstance(src, dict):
                    step_record["objects"].append(
                        {**src, "zone": "PendingCards", "position_in_zone": slot}
                    )

        # Set commit metadata used by history_features.
        if shape in {"single_ptr", "chained_cards"}:
            from family_map import ITEM_ZONE_FOR_FAMILY

            step_record["target_zone"] = ITEM_ZONE_FOR_FAMILY.get(family)
            step_record["target_position"] = int(item_ptr) if item_ptr is not None else None
        elif shape == "joker_pair":
            step_record["swap_pair"] = [
                int(decision["swap_i_local"]),
                int(decision["swap_j_local"]),
            ]
        else:
            step_record["target_zone"] = None
            step_record["target_position"] = None

        self.history_buffer.append(step_record, commit_label)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _log_decision(
        self,
        snapshot: dict[str, Any],
        action_label: str,
        decision: dict[str, Any] | None,
        elapsed_ms: float,
        emission_state: str,
    ) -> None:
        record: dict[str, Any] = {
            "ts": time.time(),
            "request_id": snapshot.get("request_id"),
            "page_name": snapshot.get("page_name"),
            "n_legal": len(snapshot.get("legal_actions") or []),
            "action_label": action_label,
            "emission_state": emission_state,
            "elapsed_ms": elapsed_ms,
            "run_id": (snapshot.get("meta") or {}).get("run_id"),
            "ante": (snapshot.get("state") or {}).get("ante"),
            "round": (snapshot.get("state") or {}).get("round"),
        }
        if decision is not None:
            record["family_name"] = decision.get("family_name")
            record["decoder_shape"] = decision.get("decoder_shape")
            record["num_cards"] = decision.get("num_cards")
            record["card_ptr_local_seq"] = decision.get("card_ptr_local_seq")
            record["item_ptr_local"] = decision.get("item_ptr_local")
            record["swap_pair_local"] = [
                decision.get("swap_i_local"),
                decision.get("swap_j_local"),
            ]
            record["family_top5"] = [
                {"family": fam, "p": p} for fam, p in decision.get("family_top5", [])
            ]
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
        prev_req_id = self._last_request_id
        if req_id is not None and req_id == prev_req_id:
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
        snapshot = self.encoder.normalize_snapshot(snapshot)

        run_id = (snapshot.get("meta") or {}).get("run_id")
        if run_id != self._last_run_id:
            self.history_buffer.clear()
            self.emission.clear()
            self._last_run_id = run_id
        if req_id is not None and prev_req_id is not None and req_id < prev_req_id:
            self.history_buffer.clear()
            self.emission.clear()

        page = snapshot.get("page_name")
        legal_set = set(snapshot.get("legal_actions") or [])

        # --- Phase A: drain any pending plan first --------------------
        if self.emission.has_pending():
            queued = self.emission.pop_next(legal_set, page)
            if queued is not None:
                self._emit_label(snapshot, req_id, queued, decision=None,
                                 elapsed_ms=0.0, emission_state="queued")
                if self.emission.finished():
                    last = self.emission.last_committed()
                    if last is not None:
                        self._commit_history_for_plan(
                            last.decision_snapshot, last.decision
                        )
                    self.emission.clear()
                return True
            # Plan invalidated mid-flight (page change / next label not
            # legal). Fall through and re-query the model.

        # --- Phase B: run the model + install a fresh plan ------------
        t0 = time.perf_counter()
        decision: dict[str, Any] | None = None
        try:
            decision = self._decide_super_step(snapshot)
            expansion = expand_decision(
                family_name=decision["family_name"],
                decoder_shape=decision["decoder_shape"],
                page_name=page,
                num_cards=int(decision["num_cards"]),
                card_ptr_local_seq=decision["card_ptr_local_seq"],
                item_ptr_local=decision["item_ptr_local"],
                swap_i_local=decision["swap_i_local"],
                swap_j_local=decision["swap_j_local"],
            )
            labels = expansion.labels
        except Exception as e:
            labels = []
            print(
                f"[agent_server] decide error: {e!r} — "
                "emitting fallback so the bot doesn't freeze"
            )
            expansion = None

        # Validate the planned first label against Lua's legal set.
        first_label_ok = bool(labels) and (
            not legal_set or labels[0] in legal_set
        )
        if not first_label_ok:
            fallback = self._fallback_label(snapshot)
            self._emit_label(
                snapshot,
                req_id,
                fallback,
                decision=decision,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                emission_state="fallback",
            )
            self.emission.clear()
            # Mirror legacy single-action history append so the model has
            # at least *some* record to look back on.
            if fallback == "StartNewRun":
                self.history_buffer.clear()
            else:
                self.history_buffer.append(
                    self.encoder.build_step(snapshot), fallback
                )
            return True

        # Happy path: install the plan and emit its first label.
        assert decision is not None
        self.emission.set_plan(
            labels,
            family_name=decision["family_name"],
            decoder_shape=decision["decoder_shape"],
            page_name=page,
            decision_snapshot=dict(snapshot),
            decision=decision,
        )
        first = self.emission.pop_next(legal_set, page)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if first is None:
            # Shouldn't happen given the first_label_ok gate above; if it
            # does, fall back rather than freeze.
            fallback = self._fallback_label(snapshot)
            self._emit_label(
                snapshot, req_id, fallback, decision=decision,
                elapsed_ms=elapsed_ms, emission_state="fallback",
            )
            self.emission.clear()
            return True
        self._emit_label(
            snapshot, req_id, first,
            decision=decision,
            elapsed_ms=elapsed_ms,
            emission_state="planned",
        )

        if first == "StartNewRun":
            self.history_buffer.clear()
            self.emission.clear()
            return True

        if self.emission.finished():
            last = self.emission.last_committed()
            if last is not None:
                self._commit_history_for_plan(
                    last.decision_snapshot, last.decision
                )
            self.emission.clear()
        return True

    def _emit_label(
        self,
        snapshot: dict[str, Any],
        req_id: Any,
        label: str,
        decision: dict[str, Any] | None,
        elapsed_ms: float,
        emission_state: str,
    ) -> None:
        line = f"{req_id if req_id is not None else 0}\t{label}\n"
        atomic_write(self.action_path, line)
        self._log_decision(snapshot, label, decision, elapsed_ms, emission_state)
        if self.verbose:
            if decision is not None and emission_state == "planned":
                top = decision.get("family_top5") or []
                top_str = ", ".join(f"{l}={p:.2f}" for l, p in top[:3])
                extra = f"  shape={decision.get('decoder_shape')}  top3=[{top_str}]"
            else:
                extra = f"  ({emission_state})"
            print(
                f"[agent_server] req={req_id} page={snapshot.get('page_name')} "
                f"-> {label}{extra}  {elapsed_ms:.1f}ms"
            )

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
