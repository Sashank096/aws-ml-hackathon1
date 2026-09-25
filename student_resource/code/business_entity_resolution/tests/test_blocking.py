import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocking import _rank_candidates


class RankCandidatesTests(unittest.TestCase):
    def test_prioritizes_candidates_supported_by_multiple_methods(self):
        found = {
            ("S2", "S2-00001"): {"name_token"},
            ("S3", "S3-99999"): {"name_token", "postal_token"},
            ("S2", "S2-00002"): {"address_token"},
        }

        selected = _rank_candidates(found, 1)

        self.assertEqual([key for key, _ in selected], [("S3", "S3-99999")])

    def test_uses_candidate_key_as_deterministic_tie_breaker(self):
        found = {
            ("S3", "S3-00002"): {"name_token"},
            ("S2", "S2-00003"): {"name_token"},
            ("S2", "S2-00001"): {"address_token"},
        }

        selected = _rank_candidates(found, 2)

        self.assertEqual(
            [key for key, _ in selected],
            [("S2", "S2-00001"), ("S2", "S2-00003")],
        )


if __name__ == "__main__":
    unittest.main()
