"""Matched, opt-in SoccerNet H/A continuation experiments.

This is a new runner; historical capacity studies and the adopted package are
never modified. Only trusted local training bundles may be loaded with joblib.
There is deliberately no organizer scoring or automatic promotion command.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import random

import joblib
import numpy as np

from experiments import learned_temporal as lt, objective_losses as ol, preparation as prep
from experiments import soccernet_capacity_model as cap, supervised
from experiments import soccernet_anchor_balance as balance
from experiments import soccernet_context_pretrain as context
from experiments import soccernet_gap_consistency as gap
from experiments.soccernet_capacity import write_once

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("control", "balanced_anchors", "gap_consistency", "context_reconstruction")
FOLDS = ("leave_out_sportec", "leave_out_metrica", "leave_out_skillcorner")
SEEDS = (2026, 2027)
TRAIN_FIXTURES = tuple(str(i) for i in (*range(3812, 3841), *range(10502, 10518)))
CONFIG = dict(updates=512, batch_size=32, lr=1e-4, gar_fraction=0.5,
              reconstruction_weight=0.05, decoder_switch_cost=3)


def _check_training(initial, teacher, old_records, gar_records, plan, fixtures):
    """Fail closed on identity, provider, target, schema and teacher mismatch."""
    names, median, scale, _ = cap._model_input_width(teacher)
    if (initial.get("kind") != "soccernet_capacity" or initial.get("hidden") != 96
            or initial.get("source", {}).get("checkpoint_sha256") != cap._checkpoint_sha(teacher)
            or list(initial.get("feature_names", [])) != names
            or not np.array_equal(initial.get("median"), median)
            or not np.array_equal(initial.get("scale"), scale)):
        raise ValueError("Continuation requires the width-96 head and its exact frozen teacher")
    expected = {(i["source"], i["match"]) for i in plan["train"]}
    actual = {cap._validate_record(r, names) for r in old_records}
    if not expected or actual != expected:
        raise ValueError("Original training records must exactly cover the whole-match TRAIN plan")
    if actual & {(i["source"], i["match"]) for i in plan["selection"]}:
        raise ValueError("Selection identity entered training")
    excluded = plan["fold"].removeprefix("leave_out_")
    if plan["fold"] not in FOLDS or any(s == excluded for s, _ in actual):
        raise ValueError("Expected a full-provider holdout")
    if not fixtures or not gar_records:
        raise ValueError("Explicit SoccerNet TRAIN fixtures and records are required")
    seen = set()
    for record in gar_records:
        source, match = cap._validate_record(record, names)
        if source != "soccernet" or match not in {f"soccernet_{f}" for f in fixtures}:
            raise ValueError("SoccerNet fixture is outside TRAIN")
        if not np.isin(record["y"], (-1, 0, 1)).all():
            raise ValueError("SoccerNet supplies sparse H/A anchors, never D labels")
        identity = (match, str(record.get("segment_id", "")))
        if not identity[1] or identity in seen:
            raise ValueError("Missing or duplicate SoccerNet segment identity")
        seen.add(identity)
    return names, median, scale


def train(initial, teacher, old_records, gar_records, *, plan, fixtures,
          arm, seed, updates=512, batch_size=32):
    """Continue one head with matched supervised draws and an isolated mechanism.

    The separate teacher object is read-only and supplies D at inference through
    ``soccernet_capacity_model.predict``. Auxiliary RNGs do not change supervised
    window draws or dropout. Extra auxiliary forward passes do add compute.
    """
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("Unknown arm or seed")
    if (isinstance(updates, bool) or int(updates) != updates or updates < 1
            or isinstance(batch_size, bool) or int(batch_size) != batch_size
            or batch_size < 2 or batch_size % 2):
        raise ValueError("Positive integer updates and even batch size >=2 required")
    names, median, scale = _check_training(initial, teacher, old_records, gar_records, plan, fixtures)
    old_groups, _ = cap._window_index(old_records, names, old=True)
    new_groups, _ = cap._window_index(gar_records, names, old=False)
    if not old_groups or not new_groups:
        raise ValueError("Both sources need supervised windows")
    spells, pairs, pair_weights, _ = cap._old_objective_targets(old_records, require_full_native=True)
    # Every arm must prove that H/A targets are supported by sparse events.
    anchor_weights, weight_receipt = balance.build_anchor_weights(gar_records, fixtures)
    weights = anchor_weights if arm == "balanced_anchors" else None
    if arm == "gap_consistency":
        for r in gar_records:
            if (np.shape(r.get("gap_x")) != np.shape(r["x"])
                    or np.shape(r.get("consistency_mask")) != np.shape(r["y"])
                    or r.get("gap_provenance", {}).get("recipe") != gap.RECIPE_VERSION):
                raise ValueError("Use attach_consistency_view on raw segments before gap training")
            if np.isinf(r["gap_x"]).any():
                raise ValueError("Gap features contain infinity")

    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net = cap.network(2 * len(names), 96)
    net.load_state_dict({k: torch.as_tensor(v).clone() for k, v in initial["state"].items()})
    frozen = copy.deepcopy(net).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    # Initialization of an auxiliary head must not perturb supervised dropout.
    with torch.random.fork_rng():
        torch.manual_seed(seed + 9000)
        head = context.reconstruction_head(96, len(names)) if arm == "context_reconstruction" else None
    parameters = list(net.parameters()) + (list(head.parameters()) if head is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=CONFIG["lr"], weight_decay=cap.WEIGHT_DECAY)
    rng, auxiliary_rng = random.Random(seed), random.Random(seed + 10000)
    auxiliary_groups = {}
    for ri, r in enumerate(gar_records):
        auxiliary_groups.setdefault(str(r["match"]), []).extend(
            (ri, start, size, full, False) for start, size, full in cap._record_chunks(r))
    history = []
    draws = []
    net.train()
    for step in range(1, updates + 1):
        bx, by, bs, bt, bm, bw, ba = [], [], [], [], [], [], []
        for is_old, count in ((True, batch_size // 2), (False, batch_size // 2)):
            for _ in range(count):
                ri, start, size, full, alternate = (cap._draw_old(rng, old_groups, old_records)
                    if is_old else cap._draw_new(rng, new_groups))
                r = (old_records if is_old else gar_records)[ri]
                swap = False if is_old else rng.random() < 0.5
                x, y = cap._prepare_window(r, start, size, full, alternate,
                                           median=median, scale=scale, swap=swap)
                bx.append(x); by.append(y)
                s = np.ones(cap.CORE, np.float32)
                t = np.zeros(cap.CORE, np.float32)
                m = np.zeros(cap.CORE, bool)
                w = np.zeros(cap.CORE, np.float32)
                a = np.ones(cap.CORE, np.float32)
                if is_old:
                    s[:size] = spells[ri][start:start + size]
                    t[:size] = pairs[ri]["live_target"][start:start + size]
                    m[:size] = pairs[ri]["live_mask"][start:start + size]
                    w[:size] = pair_weights[ri][start:start + size]
                elif weights is not None:
                    a[:size] = weights[ri][start:start + size]
                bs.append(s); bt.append(t); bm.append(m); bw.append(w); ba.append(a)
                draws.append((step, is_old, int(ri), int(start), int(size), bool(alternate), bool(swap)))
        tensor = lambda value, dtype=torch.float32: torch.as_tensor(np.asarray(value), dtype=dtype)
        logits = net(tensor(bx))
        core = logits[:, cap.CONTEXT:cap.CONTEXT + cap.CORE]
        y = tensor(by, torch.long)
        labelled = (y == 0) | (y == 1)
        if torch.any(labelled.sum(1) == 0):
            raise ValueError("Supervised draw has no anchors")
        ce = F.cross_entropy(core.transpose(1, 2), y.clamp(0, 1), reduction="none")
        window_loss = (ce * labelled * tensor(bs) * tensor(ba)).sum(1) / labelled.sum(1)
        q_left = torch.softmax(core, -1)
        q_right = torch.softmax(logits[:, cap.CONTEXT + 1:cap.CONTEXT + cap.CORE + 1], -1)
        p_change = q_left[..., 0] * q_right[..., 1] + q_left[..., 1] * q_right[..., 0]
        edge = F.binary_cross_entropy(p_change.clamp(1e-6, 1 - 1e-6), tensor(bt), reduction="none")
        transition = ol.group_balanced_binary_loss(edge, tensor(bt), tensor(bm, torch.bool), tensor(bw))
        supervised_loss = window_loss.mean() + 0.5 * cap.LIVE_TRANSITION_COEFFICIENT * (
            transition if transition is not None else core.sum() * 0)
        auxiliary_loss = core.sum() * 0
        if arm in ("gap_consistency", "context_reconstruction"):
            ax, ag, am = [], [], []
            for _ in range(batch_size // 2):
                ri, start, size, full, _ = cap._draw_new(auxiliary_rng, auxiliary_groups)
                r = gar_records[ri]
                swap = auxiliary_rng.random() < 0.5
                raw = supervised.swap_team_features(r["x"], names) if swap else r["x"]
                ax.append(lt.normalize(raw[full], median, scale))
                valid = np.zeros(len(full), bool)
                valid[cap.CONTEXT:cap.CONTEXT + size] = True
                if arm == "gap_consistency":
                    raw_gap = supervised.swap_team_features(r["gap_x"], names) if swap else r["gap_x"]
                    ag.append(lt.normalize(raw_gap[full], median, scale))
                    valid &= np.asarray(r["consistency_mask"], bool)[full]
                am.append(valid)
            # Give every arm the identical next supervised dropout RNG state.
            with torch.random.fork_rng():
                torch.manual_seed(seed + 20000 + step)
                if arm == "gap_consistency":
                    with torch.no_grad():
                        clean_logits = frozen(tensor(ax))
                    auxiliary_loss = gap.consistency_loss(clean_logits, net(tensor(ag)), tensor(am, torch.bool))
                else:
                    masked, target, mask = context.mask_context(tensor(ax), tensor(am, torch.bool), seed=seed + step)
                    auxiliary_loss = CONFIG["reconstruction_weight"] * context.masked_reconstruction_loss(
                        net, head, masked, target, mask)
        loss = supervised_loss + auxiliary_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite continuation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, cap.GRADIENT_CLIP)
        optimizer.step()
        if step == 1 or step % 32 == 0 or step == updates:
            history.append(dict(step=step, supervised=float(supervised_loss.detach()),
                                auxiliary=float(auxiliary_loss.detach()), total=float(loss.detach())))
    result = copy.deepcopy(initial)
    result.pop("continuation", None)  # old optimizer state no longer describes this fit
    result["state"] = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
    result["training"] = dict(initial.get("training", {}), input_dropout=cap.INPUT_DROPOUT)
    result["training"]["followup"] = dict(CONFIG, arm=arm, seed=seed, updates=updates,
        batch_size=batch_size, plan_sha256=prep.fingerprint(plan), fixtures=sorted(fixtures),
        initial_checkpoint_content_sha256=cap._checkpoint_sha(initial),
        supervised_draws_sha256=prep.fingerprint(draws), auxiliary_extra_compute=True if head or arm == "gap_consistency" else False,
        history=history, anchor_weights=weight_receipt)
    result["step"] = int(initial.get("step", 0)) + updates
    return result


def source_seals():
    # Include transitive experiment code, so edits invalidate a frozen run.
    paths = sorted((ROOT / "experiments").glob("*.py"))
    paths += [ROOT / "evaluation/possession_metrics.py", ROOT / "experiments/eval_set/protocol.py",
              ROOT / "experiments/eval_set/manifest.json", ROOT / "experiments/baseline_manifest.json",
              ROOT / "experiments/model_decision_policy.json",
              ROOT / "experiments/results/soccernet_capacity_declaration.json",
              ROOT / "experiments/results/soccernet_capacity_artifacts.json"]
    return {str(p.relative_to(ROOT)): prep.sha256(p) for p in paths}


def verify_checkpoint_provenance(bundle, plan):
    """Bind checkpoint pairs to their historical full-provider fit receipts."""
    fold, seed = bundle["fold"], bundle["seed"]
    historical = json.loads((ROOT / "experiments/results/soccernet_capacity_declaration.json").read_text())
    expected = historical["initial_checkpoint_provenance"]["fold_checkpoints"][f"{fold}:{seed}"]
    teacher_path, initial_path = Path(bundle["teacher_path"]), Path(bundle["initial_path"])
    if (prep.sha256(teacher_path) != expected["model_sha256"]
            or prep.sha256(teacher_path.with_suffix(".json")) != expected["receipt_sha256"]
            or set(expected["training_matches"]) != {i["match"] for i in plan["train"]}
            or set(expected["sources"]) != {i["source"] for i in plan["train"]}):
        raise ValueError("Original teacher does not have the declared full-provider provenance")
    receipt_path = initial_path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    artifacts = json.loads((ROOT / "experiments/results/soccernet_capacity_artifacts.json").read_text())
    # The sibling receipt alone is not evidence of historical checkpoint identity.
    # Bind both files to the publicly hash-pinned private integrity inventory.
    inventory_path = Path(bundle["capacity_inventory_path"])
    if prep.sha256(inventory_path) != artifacts["inventory_sha256"]:
        raise ValueError("Capacity artifact inventory differs from the published seal")
    inventory = json.loads(inventory_path.read_text())
    entries = inventory if isinstance(inventory, list) else inventory.get("files", inventory.get("artifacts", []))
    if isinstance(entries, dict):
        entries = [dict(path=k, **(v if isinstance(v, dict) else dict(sha256=v)))
                   for k, v in entries.items()]
    stage = "coarse" if seed == 2026 else "confirm"
    suffix = f"soccernet_capacity_v1/{stage}/w96_lr3e-4_step2048/{fold}/seed{seed}"
    for path, extension in ((initial_path, ".joblib"), (receipt_path, ".json")):
        matches = [e for e in entries if str(e.get("path", "")).endswith(suffix + extension)]
        if len(matches) != 1 or matches[0].get("sha256") != prep.sha256(path):
            raise ValueError("Capacity checkpoint/receipt is absent from the sealed inventory")
    checks = dict(study_id="soccernet_capacity_v1", fold=fold, seed=seed, step=2048,
        stage="coarse" if seed == 2026 else "confirm", recipe="w96_lr3e-4_step2048",
        teacher_sha256=expected["model_sha256"],
        declaration_sha256=artifacts["frozen_declaration_sha256"], model_sha256=prep.sha256(initial_path))
    if any(receipt.get(k) != v for k, v in checks.items()):
        raise ValueError("Initial capacity checkpoint has the wrong full-provider fit receipt")
    for key, path in (("teacher", teacher_path), ("initial", initial_path)):
        if cap._checkpoint_sha(joblib.load(path)) != cap._checkpoint_sha(bundle[key]):
            raise ValueError("Embedded checkpoint differs from its verified source file")
    return {str(p.resolve()): prep.sha256(p) for p in
            (teacher_path, teacher_path.with_suffix(".json"), initial_path, receipt_path, inventory_path)}


def freeze(bundle_path, output):
    """Seal a trusted, locally prepared one-fold/seed training bundle before fit."""
    bundle_path = Path(bundle_path).resolve()
    bundle = joblib.load(bundle_path)
    fold, seed = bundle["fold"], bundle["seed"]
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Unknown fold/seed")
    plan = prep.split_plan(fold=fold)
    if bundle.get("gar_split") != "train" or sorted(bundle["fixtures"]) != sorted(TRAIN_FIXTURES):
        raise ValueError("Require exactly the 45 declared official TRAIN fixtures")
    if {str(r["match"]) for r in bundle["gar_records"]} != {f"soccernet_{i}" for i in TRAIN_FIXTURES}:
        raise ValueError("The full 45-fixture GAR cohort must be present")
    _check_training(bundle["initial"], bundle["teacher"], bundle["old_records"], bundle["gar_records"], plan, bundle["fixtures"])
    balance.build_anchor_weights(bundle["gar_records"], bundle["fixtures"])
    checkpoint_files = verify_checkpoint_provenance(bundle, plan)
    if (bundle["initial"].get("training", {}).get("seed") != seed
            or bundle["teacher"].get("seed") != seed):
        raise ValueError("Bundle seed differs from its checkpoint pair")
    declaration = dict(study_id="soccernet_training_ideas_v1", bundle=str(bundle_path),
        bundle_sha256=prep.sha256(bundle_path), fold=fold, seed=seed, plan=plan,
        arms=list(ARMS), config=CONFIG, sources=source_seals(), checkpoint_files=checkpoint_files,
        versions={p: importlib.metadata.version(p) for p in ("numpy", "torch", "joblib")},
        evaluation=dict(views=["clean", "feed_like"], primary="spell_f1", tie_break="macro_f1",
            clean_screen=0.01, feed_like_floor=-0.01, provider_clean_macro_and_d_floor=-0.02,
            compare_to=["adopted_width96", "matched_continuation_control"],
            seeds=list(SEEDS), folds=list(FOLDS), actual_probability_ensemble=True),
        organizer_labels_read=False, automatic_promotion=False)
    write_once(Path(output), declaration)
    write_once(Path(str(output) + ".seal.json"), dict(declaration_sha256=prep.sha256(output)))
    return declaration


def fit(declaration_path, arm, output):
    declaration = json.loads(Path(declaration_path).read_text())
    seal = json.loads(Path(str(declaration_path) + ".seal.json").read_text())
    if seal != dict(declaration_sha256=prep.sha256(declaration_path)):
        raise ValueError("Frozen declaration bytes changed")
    if (declaration["fold"] not in FOLDS or declaration["seed"] not in SEEDS
            or declaration["plan"] != prep.split_plan(fold=declaration["fold"])
            or declaration["arms"] != list(ARMS) or arm not in ARMS):
        raise ValueError("Frozen fold, seed, plan or arms changed")
    if declaration["sources"] != source_seals() or declaration["config"] != CONFIG:
        raise ValueError("Frozen source/config changed; do not reuse this declaration")
    if declaration["versions"] != {p: importlib.metadata.version(p) for p in declaration["versions"]}:
        raise ValueError("Frozen dependency versions changed")
    if prep.sha256(declaration["bundle"]) != declaration["bundle_sha256"]:
        raise ValueError("Training bundle checksum differs")
    if any(prep.sha256(path) != digest for path, digest in declaration["checkpoint_files"].items()):
        raise ValueError("Historical checkpoint or receipt changed")
    output = Path(output)
    if output.exists() or output.with_suffix(".receipt.json").exists():
        raise ValueError("Refusing to overwrite an experiment checkpoint")
    bundle = joblib.load(declaration["bundle"])
    result = train(bundle["initial"], bundle["teacher"], bundle["old_records"], bundle["gar_records"],
        plan=declaration["plan"], fixtures=bundle["fixtures"], arm=arm, seed=declaration["seed"],
        updates=CONFIG["updates"], batch_size=CONFIG["batch_size"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        joblib.dump(result, stream)
    write_once(output.with_suffix(".receipt.json"), dict(model_sha256=prep.sha256(output),
        declaration_sha256=prep.sha256(declaration_path), arm=arm, fold=declaration["fold"],
        seed=declaration["seed"], organizer_labels_read=False))
    return result["training"]["followup"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--bundle", required=True, help="Trusted local joblib training bundle")
    f.add_argument("--output", required=True)
    r = sub.add_parser("fit")
    r.add_argument("--declaration", required=True)
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--output", required=True)
    sub.add_parser("plan")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.bundle, args.output)
    elif args.command == "fit":
        result = fit(args.declaration, args.arm, args.output)
    else:
        result = dict(arms=ARMS, config=CONFIG, folds=FOLDS, seeds=SEEDS, fits=24,
            baseline="joint_gru_soccernet_capacity_ensemble_v1", status="implementation_only",
            dependencies=["six sealed width96 full-provider heads", "matching original teachers",
                "verified original-data paired training records", "SoccerNet TRAIN merged segments",
                "raw continuous segments for paired ball-gap feature preparation"],
            automatic_promotion=False, organizer_scoring=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
