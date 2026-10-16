"""Check matrix conventions, XML decoding and repeated assembly occurrences."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/urdf_pipeline"))
from native import expand_tree, parse_transform, parse_xml, resolve_reference


class NativeTests(unittest.TestCase):
    def test_row_vector_transform_becomes_column_vector(self):
        # +90 degrees about Z, then translation (1, 2, 3) meters.
        parsed = parse_transform("0 1 0 0 -1 0 0 0 0 0 1 0 1 2 3 1")
        np.testing.assert_allclose(np.array(parsed["matrix_m"]) @ [2, 0, 0, 1], [1, 4, 3, 1])
        self.assertTrue(parsed["rigid_valid"])

    def test_invalid_affine_layout_rejected(self):
        with self.assertRaisesRegex(ValueError, "not affine"):
            parse_transform("1 0 0 2 0 1 0 0 0 0 1 0 0 0 0 1")
        self.assertFalse(parse_transform("2 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1")["rigid_valid"])

    def test_reused_subassembly_is_expanded_per_occurrence(self):
        payload = '''<?xml version="1.0" encoding="UTF-8"?><swSolidWorks xmlns="test">
          <swFile id="f" swPath="D:\\旧位置\\机械臂\\侧板.SLDPRT" swDocType="PART"/>
          <swModel id="root"><swReference id="a" swModelRef="sub" swTransform="1 0 0 0 0 1 0 0 0 0 1 0 1 0 0 1"/>
          <swReference id="b" swModelRef="sub" swSuppressed="YES" swTransform="0 1 0 0 -1 0 0 0 0 0 1 0 0 2 0 1"/></swModel>
          <swModel id="sub"><swReference id="c" swComponentName="侧板&amp;轴" swModelRef="part" swTransform="1 0 0 0 0 1 0 0 0 0 1 0 0.1 0 0 1"/></swModel>
          <swModel id="part" swFileRef="f"/><swConfiguration swModelRef="root" swMostRecentConfiguration="YES"/>
        </swSolidWorks>'''.encode("utf-8")
        instances = expand_tree(parse_xml(payload), [{"path": "机械臂/侧板.SLDPRT", "sha256": "abc"}])
        by_id = {r["instance_id"]: r for r in instances}
        self.assertEqual(by_id["a/c"]["name"], "侧板&轴")
        np.testing.assert_allclose(np.array(by_id["a/c"]["matrix_mm"])[:3, 3], [1100, 0, 0])
        np.testing.assert_allclose(np.array(by_id["b/c"]["matrix_mm"])[:3, 3], [0, 2100, 0])
        self.assertTrue(by_id["b/c"]["suppressed_in_tree"])

    def test_suffix_resolution_preserves_ambiguity(self):
        records = [{"path": "a/侧板.SLDPRT", "sha256": "a"}, {"path": "b/侧板.SLDPRT", "sha256": "b"}]
        self.assertEqual(resolve_reference("D:\\gone\\侧板.SLDPRT", records)["status"], "ambiguous")
        self.assertEqual(resolve_reference("D:\\gone\\a\\侧板.SLDPRT", records)["path"], "a/侧板.SLDPRT")


if __name__ == "__main__":
    unittest.main()
