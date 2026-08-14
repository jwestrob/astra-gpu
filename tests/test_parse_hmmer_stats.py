import unittest

from scripts.parse_hmmer_stats import parse


EXAMPLE = """\
Internal pipeline statistics summary:
-------------------------------------
Query model(s):                            1  (149 nodes)
Target sequences:                         45  (6519 residues searched)
Passed MSV filter:                        45  (1); expected 0.9 (0.02)
Passed bias filter:                       44  (0.9778); expected 0.9 (0.02)
Passed Vit filter:                        40  (0.8889); expected 0.0 (0.001)
Passed Fwd filter:                        39  (0.8667); expected 0.0 (1e-05)
Initial search space (Z):                 45  [actual number of targets]
Domain search space  (domZ):              39  [number of targets reported over threshold]
# CPU time: 0.02u 0.00s 00:00:00.02 Elapsed: 00:00:00.02
# Mc/sec: 48.32
//
"""


class ParseHmmerStatsTests(unittest.TestCase):
    def test_parses_summary(self) -> None:
        summaries = parse(EXAMPLE)
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary["inputs"]["query_models"]["nodes"], 149)
        self.assertEqual(summary["inputs"]["target_sequences"]["residues_searched"], 6519)
        self.assertEqual(summary["stages"]["bias"]["passed"], 44)
        self.assertEqual(summary["search_space"]["domZ"]["value"], 39)
        self.assertEqual(summary["reported_timing"]["elapsed_hms"], "00:00:00.02")
        self.assertEqual(summary["mc_per_second"], 48.32)

    def test_parses_multiple_summaries(self) -> None:
        self.assertEqual(len(parse(EXAMPLE + EXAMPLE)), 2)


if __name__ == "__main__":
    unittest.main()
