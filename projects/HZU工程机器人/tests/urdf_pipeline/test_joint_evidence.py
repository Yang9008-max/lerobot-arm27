"""Independent line geometry expectations for inferred mechanical axes."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/urdf_pipeline"))
from joint_evidence import line_relation, transform_axis


class JointEvidenceTests(unittest.TestCase):
    def test_parallel_opposite_axes_with_axial_offset(self):
        a = {"point_mm": [0, 10, 175], "axis": [0, 1, 0]}
        b = {"point_mm": [0, -70, 175], "axis": [0, -1, 0]}
        self.assertAlmostEqual(line_relation(a, b)["line_distance_mm"], 0)
        b["point_mm"][0] = 2.5
        self.assertAlmostEqual(line_relation(a, b)["line_distance_mm"], 2.5)

    def test_intersecting_and_skew_wrist_axes(self):
        a = {"point_mm": [-650, 0, 175], "axis": [0, 1, 0]}
        b = {"point_mm": [-675, 0, 175], "axis": [1, 0, 0]}
        self.assertAlmostEqual(line_relation(a, b)["line_distance_mm"], 0)
        self.assertAlmostEqual(line_relation(a, b)["absolute_axis_dot"], 0)
        b["point_mm"][2] = 180
        self.assertAlmostEqual(line_relation(a, b)["line_distance_mm"], 5)

    def test_axis_transform_translates_point_not_direction(self):
        matrix = [[0, -1, 0, 100], [1, 0, 0, 200], [0, 0, 1, 300], [0, 0, 0, 1]]
        result = transform_axis(matrix, [10, 0, 0], [1, 0, 0], {})
        np.testing.assert_allclose(result["point_mm"], [100, 210, 300])
        np.testing.assert_allclose(result["axis"], [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
