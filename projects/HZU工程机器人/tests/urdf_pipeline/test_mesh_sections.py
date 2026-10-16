"""Analytic radial shapes provide independent gear-period measurement expectations."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/urdf_pipeline"))
from mesh_sections import periodicity, radial_profile, section_segments


class MeshSectionTests(unittest.TestCase):
    def test_seventeen_lobed_prism_measured_without_expected_order(self):
        angle = np.arange(2048)*2*np.pi/2048
        radius = 10 + .4*np.cos(17*angle)
        a = np.stack([np.zeros_like(angle), radius*np.cos(angle), radius*np.sin(angle)], axis=1)
        b = a.copy()
        b[:, 0] = 5
        triangles = []
        for i in range(len(a)):
            j = (i+1) % len(a)
            triangles.extend([[a[i], a[j], b[j]], [a[i], b[j], b[i]]])
        profile = radial_profile(section_segments(np.array(triangles), 2.317))
        measured = periodicity(profile)
        self.assertEqual(measured["dominant_order"], 17)
        self.assertGreater(measured["dominant_energy_fraction"], .99)
        self.assertAlmostEqual(measured["radial_peak_to_peak_mm"], .8, places=4)

    def test_regular_circle_is_not_strong_tooth_evidence(self):
        measured = periodicity(np.full(2048, 12.5))
        self.assertEqual(measured["radial_peak_to_peak_mm"], 0)
        self.assertEqual(measured["dominant_energy_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
