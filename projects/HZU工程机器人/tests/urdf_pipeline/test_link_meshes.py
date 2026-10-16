"""Independent expectations for local mesh coordinates, units, and serialization."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/urdf_pipeline"))
from link_mesh_io import clean_degenerate_triangles, decode_obj, decode_stl, encode_obj, encode_stl, link_frames, pose, transform_triangles


class LinkMeshTests(unittest.TestCase):
    def test_exact_degenerate_and_float32_collapse_are_traced_without_area_threshold(self):
        source = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                           [[0, 0, 0], [1e-10, 0, 0], [0, 1e-10, 0]]])
        local = source + .1
        kept, removed = clean_degenerate_triangles(local, source)
        self.assertEqual(len(kept), 1)
        self.assertEqual(removed["source_zero_area"], [1])
        self.assertEqual(removed["stl_float32_collapse"], [2])
        # A microscopic but representable triangle at the origin is retained.
        kept_origin, _ = clean_degenerate_triangles(source[2:], source[2:])
        self.assertEqual(len(kept_origin), 1)

    def test_known_part_rotation_and_link_translation_in_meters(self):
        part = np.array([[[0, 0, 0], [.01, 0, 0], [0, .02, 0]]])
        part_in_arm = pose([.1, .2, .3], [0, 0, np.pi/2])
        link_in_arm = pose([.05, .1, .2], [np.pi/2, 0, 0])
        local = transform_triangles(part, np.linalg.inv(link_in_arm) @ part_in_arm)
        np.testing.assert_allclose(local, [[[.05, .1, -.1], [.05, .1, -.11], [.03, .1, -.1]]], atol=1e-14)
        np.testing.assert_allclose(transform_triangles(local, link_in_arm),
                                   [[[.1, .2, .3], [.1, .21, .3], [.08, .2, .3]]], atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(local[0, 1]-local[0, 0]), .01)

    def test_file_roundtrip_preserves_triangle_order_and_recomputed_normals(self):
        triangles = np.array([[[0, 0, 0], [0, 0, .01], [0, .02, 0]],
                              [[.1, 0, 0], [.11, 0, 0], [.1, 0, .03]]])
        vertices, normals = decode_stl(encode_stl(triangles))
        np.testing.assert_allclose(vertices, triangles, atol=1e-8)
        np.testing.assert_allclose(normals, [[-1, 0, 0], [0, -1, 0]], atol=1e-8)
        np.testing.assert_allclose(decode_obj(encode_obj(triangles)), triangles, atol=1e-12)

    def test_reflection_scale_and_invalid_binary_are_rejected(self):
        triangle = np.zeros((1, 3, 3))
        for value in [-1, .001]:
            matrix = np.eye(4)
            matrix[0, 0] = value
            with self.assertRaises(ValueError):
                transform_triangles(triangle, matrix)
        with self.assertRaises(ValueError):
            decode_stl(encode_stl(triangle)[:-1])

    def test_two_joint_frame_chain_known_quarter_turn(self):
        config = {"joints": [{"parent": "root", "child": "a", "origin_xyz_m": [.1, 0, 0],
                              "origin_rpy_rad": [0, 0, np.pi/2], "axis": [0, 0, 1]},
                             {"parent": "a", "child": "b", "origin_xyz_m": [.2, 0, 0],
                              "origin_rpy_rad": [0, 0, 0], "axis": [0, 1, 0]}]}
        frames = link_frames(config)
        np.testing.assert_allclose(frames["b"][:3, 3], [.1, .2, 0], atol=1e-14)
        config["joints"][0]["parent"] = "b"
        with self.assertRaises(ValueError):
            link_frames(config)


if __name__ == "__main__":
    unittest.main()
