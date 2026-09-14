"""Standard-library checks for reference trace segmentation and statistics."""
import csv
import hashlib
import io
import math
import unittest

from tools.analyze_reference_trace import (
    COMMAND_COLUMNS, DT, REQUIRED, analyze_trace, moments, parse_trace, pool_moments,
)


def trace_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def sample(step, episode_step, *, timeout=False, x=0.0):
    row = dict.fromkeys(REQUIRED, 0)
    row.update(step=step, policy_tick=step, sim_s=step * DT,
               episode_step=episode_step, episode_time_s=episode_step * DT,
               sample_kind="pre_reset", terminated=False, timeout=timeout,
               diagnostic_non_wheel_contact=False, z=0.3, x=x)
    for name in COMMAND_COLUMNS:
        row[name] = 0.3 if name.endswith("height") else 0
    return row


class ReferenceTraceTest(unittest.TestCase):
    def test_signed_variance_and_unequal_episode_pooling(self):
        a, b = [-1.0, 1.0], [10.0, 10.0, 10.0, 10.0]
        mean, within, between = pool_moments([(len(v), *moments(v)) for v in (a, b)])
        self.assertAlmostEqual(mean, 20 / 3)
        self.assertAlmostEqual(within, 1 / 3)
        self.assertAlmostEqual(moments(a + b)[1], within + between)
        self.assertEqual(moments([abs(x) for x in a])[1], 0)
        self.assertEqual(moments(a)[1], 1)

    def test_reset_terminal_ownership_and_warmup_each_episode(self):
        rows = [sample(i, i, timeout=i == 202, x=float(i)) for i in range(1, 203)]
        rows += [sample(202 + i, i, x=-1000 + float(i)) for i in range(1, 204)]
        data = trace_bytes(rows)
        result = analyze_trace(data, 0.3, hashlib.sha256(data).hexdigest(), len(rows))
        first, second = result["episodes"]
        self.assertEqual([first["steady_samples"], second["steady_samples"]], [2, 3])
        self.assertEqual(first["steady_steps"], [201, 202])
        self.assertEqual(second["steady_steps"], [403, 405])
        self.assertEqual(first["all_xy"][2], 201)
        self.assertEqual(second["all_xy"][2], 202)
        self.assertEqual(second["steady_xy"][2], 2)
        self.assertTrue(first["timeout"])

    def test_reject_bad_rows_instead_of_silently_dropping(self):
        mutations = [
            {"vx": math.nan}, {"action_cmd_vx": 1}, {"reward_cmd_height": 0.31},
            {"step": 3}, {"episode_step": 1}, {"sim_s": 0.021},
            {"episode_time_s": 0.03}, {"sample_kind": "post_reset"},
            {"timeout": "unknown"}, {"diagnostic_non_wheel_contact": True},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                rows = [sample(1, 1), sample(2, 2)]
                rows[1].update(mutation)
                with self.assertRaisesRegex(ValueError, "CSV line 3"):
                    parse_trace(trace_bytes(rows), 0.3)

    def test_reject_hash_mismatch_and_blank_rows(self):
        data = trace_bytes([sample(1, 1), sample(2, 2)])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            analyze_trace(data, 0.3, "0" * 64)
        with self.assertRaisesRegex(ValueError, "blank"):
            parse_trace(data + b"\n", 0.3)


if __name__ == "__main__":
    unittest.main()
