import unittest

from playlist import _resolve

TRACKS = [
    {"order_index": 0, "duration_seconds": 100.0, "cumulative_start_seconds": 0.0},
    {"order_index": 1, "duration_seconds": 150.0, "cumulative_start_seconds": 100.0},
    {"order_index": 2, "duration_seconds": 80.0, "cumulative_start_seconds": 250.0},
]


class TestResolvePosition(unittest.TestCase):
    def test_start_of_story(self):
        self.assertEqual(_resolve(TRACKS, 0), (0, 0.0))

    def test_middle_of_first_track(self):
        self.assertEqual(_resolve(TRACKS, 50), (0, 50.0))

    def test_exact_track_boundary(self):
        self.assertEqual(_resolve(TRACKS, 100), (1, 0.0))

    def test_middle_of_second_track(self):
        self.assertEqual(_resolve(TRACKS, 200), (1, 100.0))

    def test_last_track(self):
        self.assertEqual(_resolve(TRACKS, 300), (2, 50.0))

    def test_negative_clamped_to_zero(self):
        self.assertEqual(_resolve(TRACKS, -10), (0, 0.0))

    def test_beyond_end_clamped_to_last_track_duration(self):
        self.assertEqual(_resolve(TRACKS, 10_000), (2, 80.0))

    def test_empty_tracks_raises(self):
        with self.assertRaises(ValueError):
            _resolve([], 0)


if __name__ == "__main__":
    unittest.main()
