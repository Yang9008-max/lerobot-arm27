"""3MF hierarchy and unit conversions using a tiny in-memory package."""
import io
from pathlib import Path
import sys
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/urdf_pipeline"))
from print_meshes import read_3mf


class PrintMeshTests(unittest.TestCase):
    def test_nested_rotation_translation_and_build_unit_scale(self):
        model = '''<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" unit="millimeter">
        <resources><object id="1"><mesh><vertices><vertex x="0" y="0" z="0"/><vertex x="10" y="0" z="0"/>
        <vertex x="0" y="20" z="0"/></vertices><triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>
        <object id="2"><components><component objectid="1" transform="0 1 0 -1 0 0 0 0 1 100 0 0"/>
        </components></object></resources><build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 0 30 0"/></build></model>'''
        package = io.BytesIO()
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("_rels/.rels", '<Relationships><Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" Target="/3D/3dmodel.model"/></Relationships>')
            archive.writestr("3D/3dmodel.model", model)
        triangles, metadata = read_3mf(package.getvalue())
        np.testing.assert_allclose(triangles[0], [[.1, .03, 0], [.1, .04, 0], [.08, .03, 0]])
        self.assertEqual(metadata["triangles"], 1)


if __name__ == "__main__":
    unittest.main()
