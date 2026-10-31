import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.provider_touch_feasibility import audit_sources


class ProviderTouchFeasibilityTests(unittest.TestCase):
    def _write_csv(self, path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_inventory_counts_source_native_event_coverage_without_touch_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "external" / "raw"
            self._write_csv(raw / "metrica/Sample_Game_1/Sample_Game_1_RawEventsData.csv",
                            ["Team", "Type", "Start Frame", "End Frame", "Period"],
                            [{"Team": "Home", "Type": "PASS", "Start Frame": "2",
                              "End Frame": "4", "Period": "1"},
                             {"Team": "Away", "Type": "CHALLENGE", "Start Frame": "5",
                              "End Frame": "5", "Period": "1"}])
            self._write_csv(raw / "metrica/Sample_Game_2/Sample_Game_2_RawEventsData.csv",
                            ["Team", "Type", "Start Frame", "End Frame", "Period"],
                            [{"Team": "Home", "Type": "PASS", "Start Frame": "2",
                              "End Frame": "4", "Period": "1"}])
            (raw / "metrica/Sample_Game_3").mkdir(parents=True)
            (raw / "metrica/Sample_Game_3/Sample_Game_3_events.json").write_text(json.dumps({
                "data": [{"type": {"name": "CARRY"}, "team": {"id": 1}, "start": {"frame": 2},
                          "end": {"frame": 4}, "period": 1}]}))

            sc_ids = (1874553, 1886347, 1899585, 1925299, 1927964, 1953632,
                      1959846, 1986691, 1996435, 1996436, 2006229)
            sc_fields = ["event_id", "event_type", "frame_start", "frame_end", "team_id",
                         "game_interruption_after", "game_interruption_before"]
            for mid in sc_ids:
                self._write_csv(raw / str(mid) / f"{mid}_dynamic_events.csv", sc_fields,
                                [{"event_id": "1", "event_type": "player_possession",
                                  "frame_start": "2", "frame_end": "4", "team_id": "9",
                                  "game_interruption_after": "throw_in", "game_interruption_before": ""}])
            source = lambda train, selection: {
                "train": [{"match": x} for x in train],
                "selection": [{"match": x} for x in selection],
            }
            manifest = {"sources": {
                "sportec": source(["J03WMX"], ["J03WOY"]),
                "metrica": source(["metrica_1", "metrica_2"], ["metrica_3"]),
                "skillcorner": source([f"skillcorner_had_{x}" for x in sc_ids[:8]],
                                       [f"skillcorner_had_{x}" for x in sc_ids[8:]]),
            }}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            paths = list(raw.rglob("*.csv")) + list(raw.rglob("*.json"))
            result = audit_sources(root, paths, manifest_path)

            self.assertEqual(result["metrica"]["totals_by_manifest_cohort"]["train"]["events"], 3)
            self.assertEqual(result["metrica"]["totals_by_manifest_cohort"]["selection"]["events"], 1)
            self.assertEqual(result["skillcorner"]["totals_by_manifest_cohort"]["train"]["possession_events"], 8)
            self.assertFalse(result["skillcorner"]["matches"][0]["gar_control_flag_present"])
            self.assertEqual(result["skillcorner"]["matches"][0]["possession_rows_with_interruption_after"], 1)
            self.assertFalse(result["sportec"]["local_event_stream_present"])
            self.assertIn("unsupported", result["skillcorner"]["touch_run_mapping"])
            self.assertIn("without inventing", result["decision"])


if __name__ == "__main__":
    unittest.main()
