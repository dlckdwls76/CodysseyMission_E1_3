"""Mini NPU Simulator 핵심 기능 테스트."""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import main


class MiniNPUTest(unittest.TestCase):
    def setUp(self):
        self.cross = main.generate_pattern(3, "Cross")
        self.x_pattern = main.generate_pattern(3, "X")

    def test_mac_scores(self):
        self.assertEqual(main.mac(self.cross, self.cross), 5.0)
        self.assertEqual(main.mac(self.cross, self.x_pattern), 1.0)

    def test_normalize_label(self):
        self.assertEqual(main.normalize_label("+"), "Cross")
        self.assertEqual(main.normalize_label("cross"), "Cross")
        self.assertEqual(main.normalize_label("X"), "X")

    def test_epsilon_tie_is_undecided(self):
        result = main.decide(0.9, 0.9 - 1e-10, "Cross", "X")
        self.assertEqual(result, "UNDECIDED")

    def test_validate_matrix_rejects_wrong_size(self):
        with self.assertRaises(ValueError):
            main.validate_matrix([[1, 0], [0, 1]], 3, "test")

    def test_invalid_json_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text("{invalid", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data.json 형식 오류"):
                main.load_json_data(path)

    def test_example_data_has_no_failures(self):
        data = json.loads(main.DATA_PATH.read_text(encoding="utf-8"))
        filters, errors = main.load_filters(data["filters"])
        self.assertEqual(errors, [])
        with redirect_stdout(io.StringIO()):
            total, passed, failures, _ = main.analyze_patterns(
                data["patterns"], filters
            )
        self.assertEqual(total, 6)
        self.assertEqual(passed, 6)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
