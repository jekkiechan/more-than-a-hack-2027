"""Event-schema feasibility audit for provider touch-rule calibration.

The audit inventories provider annotations; it never synthesizes GAR touch
runs, scores candidate labels, or fits a model. Event-only downloads are kept
under ignored ``data/external/raw`` and omit tracking streams.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
METRICA_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
SKILLCORNER_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
STUDY_ID = "provider_touch_feasibility_v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    temp.replace(path)


def fetch_event_only_sources(data_dir: Path) -> list[Path]:
    """Fetch only the documented Metrica and SkillCorner event files."""
    raw = data_dir / "external" / "raw"
    paths = []
    for game in (1, 2):
        filename = f"Sample_Game_{game}_RawEventsData.csv"
        path = raw / "metrica" / f"Sample_Game_{game}" / filename
        url = (f"https://raw.githubusercontent.com/metrica-sports/sample-data/{METRICA_COMMIT}"
               f"/data/Sample_Game_{game}/{filename}")
        if not path.is_file():
            _download(url, path)
        paths.append(path)
    filename = "Sample_Game_3_events.json"
    path = raw / "metrica" / "Sample_Game_3" / filename
    url = (f"https://raw.githubusercontent.com/metrica-sports/sample-data/{METRICA_COMMIT}"
           f"/data/Sample_Game_3/{filename}")
    if not path.is_file():
        _download(url, path)
    paths.append(path)

    match_ids = (1874553, 1886347, 1899585, 1925299, 1927964, 1953632,
                 1959846, 1986691, 1996435, 1996436, 2006229)
    for match_id in match_ids:
        filename = f"{match_id}_dynamic_events.csv"
        path = raw / str(match_id) / filename
        url = (f"https://raw.githubusercontent.com/SkillCorner/opendata/{SKILLCORNER_COMMIT}"
               f"/data/matches/{match_id}/{filename}")
        if not path.is_file():
            _download(url, path)
        paths.append(path)
    return paths


def _metrica_summary(path: Path, game: int) -> dict:
    if game in (1, 2):
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        fields = sorted(rows[0]) if rows else []
        types = Counter(row.get("Type", "") for row in rows)
        has_start = sum(bool(row.get("Start Frame")) for row in rows)
        has_end = sum(bool(row.get("End Frame")) for row in rows)
        has_team = sum(bool(row.get("Team")) for row in rows)
    else:
        obj = json.loads(path.read_text())
        rows = obj["data"]
        fields = sorted({key for row in rows for key in row})
        types = Counter(row.get("type", {}).get("name", "") for row in rows)
        has_start = sum("frame" in row.get("start", {}) for row in rows)
        has_end = sum("frame" in row.get("end", {}) for row in rows)
        has_team = sum(bool(row.get("team")) for row in rows)
    return {
        "match": f"metrica_{game}", "events": len(rows), "fields": fields,
        "event_types": dict(sorted(types.items())),
        "rows_with_start_frame": has_start, "rows_with_end_frame": has_end,
        "rows_with_team": has_team,
        "gar_game_event_id_present": any(k.lower() in ("game_event_id", "event_id") for k in fields),
        "gar_control_flag_present": "controlled" in {k.lower() for k in fields},
    }


def _skillcorner_summary(path: Path, match_id: int) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fields = set(rows[0]) if rows else set()
    possessions = [row for row in rows if row.get("event_type") == "player_possession"]
    return {
        "match": f"skillcorner_had_{match_id}", "events": len(rows),
        "possession_events": len(possessions),
        "event_types": dict(sorted(Counter(row.get("event_type", "") for row in rows).items())),
        "possession_rows_with_frame_start": sum(bool(row.get("frame_start")) for row in possessions),
        "possession_rows_with_frame_end": sum(bool(row.get("frame_end")) for row in possessions),
        "possession_rows_with_team": sum(bool(row.get("team_id")) for row in possessions),
        "possession_rows_with_interruption_after": sum(bool(row.get("game_interruption_after")) for row in possessions),
        "possession_rows_with_interruption_before": sum(bool(row.get("game_interruption_before")) for row in possessions),
        "gar_game_event_type_present": "game_event_type" in fields,
        "gar_game_event_id_present": "game_event_id" in fields,
        "gar_possession_event_type_present": "possession_event_type" in fields,
        "gar_control_flag_present": "controlled" in fields,
    }


def audit_sources(data_dir: Path, event_paths: list[Path] | None = None,
                  manifest_path: Path | None = None) -> dict:
    """Summarize native event coverage and explicit field incompatibilities."""
    data_dir = Path(data_dir)
    manifest_path = manifest_path or ROOT / "experiments/eval_set/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sources = manifest["sources"]
    if event_paths is None:
        event_paths = fetch_event_only_sources(data_dir)
    by_name = {p.name: p for p in event_paths}
    metrica = [_metrica_summary(by_name[f"Sample_Game_{g}_RawEventsData.csv"]
                                if g < 3 else by_name["Sample_Game_3_events.json"], g)
               for g in (1, 2, 3)]
    ids = (1874553, 1886347, 1899585, 1925299, 1927964, 1953632,
           1959846, 1986691, 1996435, 1996436, 2006229)
    skillcorner = [_skillcorner_summary(by_name[f"{mid}_dynamic_events.csv"], mid) for mid in ids]

    def cohorts(source: str) -> dict:
        entry = sources[source]
        return {split: [item["match"] for item in entry.get(split, [])]
                for split in ("train", "selection")}

    metrica_by_match = {row["match"]: row for row in metrica}
    skillcorner_by_match = {row["match"]: row for row in skillcorner}

    def totals(rows: list[dict], keys: tuple[str, ...]) -> dict:
        return {key: sum(int(row[key]) for row in rows) for key in keys}

    def cohort_totals(source: str, lookup: dict[str, dict], keys: tuple[str, ...]) -> dict:
        source_cohorts = cohorts(source)
        return {split: totals([lookup[match] for match in source_cohorts[split]
                               if match in lookup], keys)
                for split in ("train", "selection")}

    local_root = data_dir / "external"
    external_cache = local_root / "processed"
    sportec_cache = data_dir / "external_audit" / "sportec" / "processed"
    raw_sportec = data_dir / "external_audit" / "sportec" / "raw"
    files = []
    for path in sorted(event_paths):
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {
        "study_id": STUDY_ID,
        "purpose": "schema and field coverage feasibility; no labels synthesized/scored and no model fitted",
        "inputs": {
            "source_manifest": str(manifest_path), "source_manifest_sha256": _sha256(manifest_path),
            "metrica_commit": METRICA_COMMIT, "skillcorner_commit": SKILLCORNER_COMMIT,
            "event_files": files,
            "code_sha256": {name: _sha256(ROOT / "experiments" / name) for name in (
                "provider_touch_feasibility.py", "provider_touch_calibration.py",
                "soccernet_dense_labels.py", "metrica_open.py", "skillcorner_had.py",
                "audit_dataset.py")},
        },
        "cohorts": {name: cohorts(name) for name in ("sportec", "metrica", "skillcorner")},
        "metrica": {"matches": metrica,
                    "totals_by_manifest_cohort": cohort_totals(
                        "metrica", metrica_by_match,
                        ("events", "rows_with_start_frame", "rows_with_end_frame", "rows_with_team")),
                    "touch_run_mapping": "unsupported: no GAR event id or controlled flag; events are point/span actions with source-specific meanings"},
        "skillcorner": {"matches": skillcorner,
                        "totals_by_manifest_cohort": cohort_totals(
                            "skillcorner", skillcorner_by_match,
                            ("events", "possession_events", "possession_rows_with_frame_start",
                             "possession_rows_with_frame_end", "possession_rows_with_team",
                             "possession_rows_with_interruption_after",
                             "possession_rows_with_interruption_before")),
                        "touch_run_mapping": "unsupported: direct player_possession spans and interruption tags, not GAR OTB touch runs or control flags"},
        "sportec": {
            "label_source": "BallStatus/BallPossession frame annotations in DFL XML",
            "train_matches": [item["match"] for item in sources["sportec"]["train"]],
            "selection_matches": [item["match"] for item in sources["sportec"]["selection"]],
            "local_event_stream_present": any(raw_sportec.rglob("*")) if raw_sportec.exists() else False,
            "processed_cache_present": any(sportec_cache.glob("*.npz")),
            "touch_run_mapping": "unsupported: no event run IDs, event-end type, or controlled/uncontrolled event labels in the documented reference schema",
        },
        "local_raw_cache": {
            "metrica_event_dir_present": (local_root / "raw" / "metrica").exists(),
            "skillcorner_dynamic_event_files_present": sum((local_root / "raw" / str(mid) /
                f"{mid}_dynamic_events.csv").is_file() for mid in ids),
            "sportec_event_dir_present": raw_sportec.exists(),
            "provider_processed_cache_dir_present": external_cache.exists(),
            "scope": "checks current data_dir only; private mtah-data LFS refs are separate and may be pointer-only",
        },
        "decision": "No cross-provider touch-rule calibration is supportable: the matches are distinct, providers do not share GAR run-control fields, and source labels cannot be turned into paired touch annotations without inventing them.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/results/provider_touch_feasibility_v1.json")
    args = parser.parse_args()
    result = audit_sources(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"study_id": STUDY_ID, "output": str(args.output),
                      "metrica_train_events": result["metrica"]["totals_by_manifest_cohort"]["train"]["events"],
                      "skillcorner_train_possessions": result["skillcorner"]["totals_by_manifest_cohort"]["train"]["possession_events"],
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
