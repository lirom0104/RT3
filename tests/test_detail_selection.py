import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_thermal_detail import select


def row(event, step, relative, boundary):
    return {"event": event, "step": step, "metrics": {
        mode: {"abs_rel": relative, "rgb_boundary": boundary if mode != "thermal" else 0}
        for mode in ("rgb", "mixed", "thermal")}}


class DetailSelectionTests(unittest.TestCase):
    def test_boundary_and_geometry_units_do_not_control_selection(self):
        rows = [row("baseline", 0, .1, .01), row("validation", 50, .101, .008)]
        selected = select(rows)["selected"]
        self.assertEqual(selected["step"], 50)
        for r in rows:
            for m in r["metrics"].values():
                m["rgb_boundary"] *= 100
        self.assertEqual(select(rows)["selected"]["step"], 50)

    def test_rgb_regression_guard_rejects_candidate_despite_sharpness(self):
        baseline = row("baseline", 0, .1, .01)
        candidate = row("validation", 50, .1, .001)
        candidate["metrics"]["rgb"]["abs_rel"] = .12
        result = select([baseline, candidate])
        self.assertEqual(result["selected"]["step"], 0)
        self.assertFalse(result["candidates"][1]["allowed"])


if __name__ == "__main__":
    unittest.main()
