"""Known rigid rotations test the independent world-axis FK oracle."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/urdf_pipeline"))
from verify_joint_motion import rotation_about_line


class WorldAxisRotationTests(unittest.TestCase):
    def test_offset_axis_quarter_turn_and_axis_fixed_points(self):
        matrix = rotation_about_line(np.array([1., 2., 3.]), [0, 0, 1], np.pi/2)
        np.testing.assert_allclose(matrix @ [2, 2, 3, 1], [1, 3, 3, 1], atol=1e-14)
        np.testing.assert_allclose(matrix @ [1, 2, 7, 1], [1, 2, 7, 1], atol=1e-14)

    def test_axis_point_shift_and_axis_sign(self):
        p, axis = np.array([.4, -.2, .3]), np.array([1., 2., -2.])/3
        a = rotation_about_line(p, axis, .7)
        b = rotation_about_line(p + 4*axis, -axis, -.7)
        np.testing.assert_allclose(a, b, atol=1e-14)
        np.testing.assert_allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-14)


if __name__ == "__main__":
    unittest.main()
