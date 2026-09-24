import unittest

from soramimic_score.audio import MelodyNote
from soramimic_score.line_windows import snap_line_windows_to_rests


class LineWindowTests(unittest.TestCase):
    def test_adjacent_whisper_lines_meet_at_nearby_melody_rest(self):
        windows = ((0.0, 2.1), (2.3, 4.0))
        notes = (MelodyNote(0, 1.7, 60), MelodyNote(2.3, 3.8, 62))
        self.assertEqual(snap_line_windows_to_rests(windows, notes),
                         ((0.0, 2.0), (2.0, 4.0)))

    def test_distant_rest_does_not_move_whisper_lines(self):
        windows = ((0.0, 1.0), (1.1, 2.0))
        notes = (MelodyNote(0, .2, 60), MelodyNote(4.0, 5.0, 62))
        self.assertEqual(snap_line_windows_to_rests(windows, notes), windows)
