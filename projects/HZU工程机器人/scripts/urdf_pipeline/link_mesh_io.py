"""Small meter-preserving mesh operations used by the link export pipeline."""
import struct

import numpy as np


STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def require_rigid(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 4x4 transform")
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0) or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0) or abs(np.linalg.det(rotation)-1) > 1e-9:
        raise ValueError("Expected proper rigid transform; scale/reflection not accepted")
    return matrix


def transform_triangles(triangles_m, matrix_m):
    """Inputs and outputs are meters; only the caller converts CAD translations."""
    matrix = require_rigid(matrix_m)
    triangles = np.asarray(triangles_m, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError("Expected finite Nx3x3 triangle vertices")
    return triangles @ matrix[:3, :3].T + matrix[:3, 3]


def decode_stl(data):
    if len(data) < 84:
        raise ValueError("Truncated binary STL")
    count = struct.unpack_from("<I", data, 80)[0]
    if not count or len(data) != 84 + count*50:
        raise ValueError("Invalid binary STL length or triangle count")
    records = np.frombuffer(data, dtype=STL_RECORD, offset=84)
    vertices = records["vertices"].astype(float)
    if not np.isfinite(vertices).all() or not np.isfinite(records["normal"]).all():
        raise ValueError("Nonfinite STL coordinates or normals")
    return vertices, records["normal"].astype(float)


def clean_degenerate_triangles(local_triangles, source_triangles):
    """Keep faces representable in STL; record every removed source index.

    No area threshold or target face count is used. Only exactly zero cross
    products in the source, local double precision, or final float32 vertices
    are removed. OBJ uses the same retained source faces for correspondence.
    """
    local = np.asarray(local_triangles, dtype=float)
    source = np.asarray(source_triangles, dtype=float)
    if local.shape != source.shape:
        raise ValueError("Source/local triangle shapes differ")

    def zero_area(triangles):
        cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        return np.all(cross == 0, axis=1)

    quantized = local.astype("<f4").astype(float)
    if not np.isfinite(quantized).all():
        raise ValueError("Nonfinite or overflowing mesh coordinates")
    source_zero = zero_area(source)
    local_zero = zero_area(local) & ~source_zero
    float32_zero = zero_area(quantized) & ~source_zero & ~local_zero
    removed = {"source_zero_area": np.flatnonzero(source_zero).tolist(),
               "local_numeric_zero_area": np.flatnonzero(local_zero).tolist(),
               "stl_float32_collapse": np.flatnonzero(float32_zero).tolist()}
    keep = ~(source_zero | local_zero | float32_zero)
    return local[keep], removed


def encode_stl(triangles_m):
    triangles = np.asarray(triangles_m, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("Expected nonempty finite triangles")
    records = np.zeros(len(triangles), dtype=STL_RECORD)
    records["vertices"] = triangles
    if not np.isfinite(records["vertices"]).all():
        raise ValueError("STL float32 overflow")
    # Recompute normals from the actual float32 vertices being written.
    quantized = records["vertices"].astype(float)
    normals = np.cross(quantized[:, 1]-quantized[:, 0], quantized[:, 2]-quantized[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 0)
    records["normal"] = normals
    return b"HZU link visual; units=m; components concatenated".ljust(80, b"\0") + struct.pack("<I", len(records)) + records.tobytes()


def encode_obj(triangles_m):
    triangles = np.asarray(triangles_m, dtype=float)
    if not np.isfinite(triangles).all():
        raise ValueError("Nonfinite OBJ vertices")
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    lines = ["# HZU link visual; units=m; triangle order preserved"]
    lines += ["v " + " ".join(format(float(x), ".10g") for x in v) for v in vertices]
    lines += ["f " + " ".join(str(int(x)+1) for x in face) for face in inverse.reshape(-1, 3)]
    return ("\n".join(lines) + "\n").encode("ascii")


def decode_obj(data):
    vertices, faces = [], []
    for line in data.decode("ascii").splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            if len(fields) != 4:
                raise ValueError("Expected three OBJ vertex coordinates")
            vertices.append(list(map(float, fields[1:])))
        elif fields and fields[0] == "f":
            if len(fields) != 4:
                raise ValueError("Expected triangular OBJ face")
            faces.append([int(x)-1 for x in fields[1:]])
        elif fields and not fields[0].startswith("#"):
            raise ValueError("Unsupported record in generated OBJ")
    vertices, faces = np.array(vertices), np.array(faces)
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid OBJ vertices/faces")
    return vertices[faces]


def pose(xyz, rpy):
    r, p, y = np.asarray(rpy, dtype=float)
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]])
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, xyz
    return require_rigid(matrix)


def link_frames(joints):
    definitions = joints["joints"]
    children = {j["child"] for j in definitions}
    roots = {j["parent"] for j in definitions} - children
    if len(roots) != 1 or len(children) != len(definitions):
        raise ValueError("Expected one root and one parent per child")
    root = next(iter(roots))
    frames = {root: np.eye(4)}  # Root is explicitly defined in CAD frame A.
    pending = list(definitions)
    while pending:
        available = [j for j in pending if j["parent"] in frames]
        if not available:
            raise ValueError("Disconnected or cyclic joint graph")
        for joint in available:
            if any(joint.get(k) is None for k in ["origin_xyz_m", "origin_rpy_rad", "axis"]):
                raise ValueError("Joint model frame is not yet defined")
            frames[joint["child"]] = frames[joint["parent"]] @ pose(joint["origin_xyz_m"], joint["origin_rpy_rad"])
            pending.remove(joint)
    return frames
