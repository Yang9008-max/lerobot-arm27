"""Independent geometric expectations for the mm-to-m pipeline."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import cadquery as cq
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/urdf_pipeline"))
from geometry import analyze_shape, process_step, read_stl_bounds


class GeometryTests(unittest.TestCase):
    def test_box_volume_and_unit_density_inertia(self):
        result = analyze_shape(cq.Workplane("XY").box(10, 20, 30).val())
        self.assertAlmostEqual(result["volume_m3"], 6e-6, places=14)
        np.testing.assert_allclose(result["unit_density_inertia_at_centroid_m5"],
                                   np.diag([6.5e-10, 5e-10, 2.5e-10]), rtol=1e-9, atol=1e-20)

    def test_step_to_stl_keeps_origin_and_scales_once(self):
        shape = cq.Workplane("XY").box(10, 20, 30).translate((100, -40, 60)).val()
        temporary_root = Path(__file__).resolve().parents[2] / "urdf_work" / "test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            step = Path(temporary) / "box.step"
            mesh = Path(temporary) / "box.stl"
            cq.exporters.export(shape, str(step), exportType="STEP")
            result = process_step(step, mesh)
            bounds = read_stl_bounds(mesh)
            np.testing.assert_allclose(bounds["bbox_size_m"], [0.01, 0.02, 0.03], atol=1e-8)
            np.testing.assert_allclose(bounds["bbox_min_m"], [0.095, -0.05, 0.045], atol=1e-8)
            np.testing.assert_allclose(result["geometric_centroid_mm"], [100, -40, 60], atol=1e-8)

    def test_cylinder_axis_and_radius(self):
        result = analyze_shape(cq.Workplane("XY").circle(5).extrude(20).val())
        self.assertAlmostEqual(result["volume_mm3"], math.pi * 25 * 20, places=6)
        self.assertEqual(len(result["cylinder_candidates"]), 1)
        axis = result["cylinder_candidates"][0]
        np.testing.assert_allclose(axis["axis"], [0, 0, 1], atol=1e-10)
        self.assertAlmostEqual(axis["radius_mm"], 5)

    def test_surface_only_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No solid"):
            analyze_shape(cq.Workplane("XY").rect(10, 20).val())


if __name__ == "__main__":
    unittest.main()
