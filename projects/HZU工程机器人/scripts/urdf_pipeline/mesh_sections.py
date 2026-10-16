"""Independent cross-section measurements of the supplied triangulated gear meshes."""
from collections import Counter
import struct

import numpy as np


def read_binary_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50:
        raise ValueError("Unexpected binary STL length")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    return np.frombuffer(data, dtype=dtype, offset=84)["vertices"].astype(float)


def section_segments(triangles, x):
    distances = triangles[:, :, 0] - x
    mask = (distances.min(axis=1) < 0) & (distances.max(axis=1) > 0)
    selected = triangles[mask]
    if not len(selected):
        return np.empty((0, 2, 2))
    intersections = [[] for _ in selected]
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        a, b = selected[:, i], selected[:, j]
        crosses = (a[:, 0] < x) != (b[:, 0] < x)
        for k in np.flatnonzero(crosses):
            ratio = (x-a[k, 0]) / (b[k, 0]-a[k, 0])
            intersections[k].append((a[k] + ratio * (b[k]-a[k]))[1:])
    return np.array([points for points in intersections if len(points) == 2])


def radial_profile(segments, samples=2048):
    # Intersect rays against every section edge. Choosing the outermost positive
    # intersection discards bore contours and requires no guessed tooth count.
    a = segments[:, 0]
    edge = segments[:, 1] - a
    numerator = a[:, 0]*edge[:, 1] - a[:, 1]*edge[:, 0]
    angles = np.arange(samples)*2*np.pi/samples
    radii = []
    for chunk in np.array_split(angles, 16):
        dx, dy = np.cos(chunk), np.sin(chunk)
        denominator = dx[None, :]*edge[:, 1, None] - dy[None, :]*edge[:, 0, None]
        safe = np.where(abs(denominator) > 1e-12, denominator, np.nan)
        r = numerator[:, None] / safe
        u = (a[:, 0, None]*dy[None, :] - a[:, 1, None]*dx[None, :]) / safe
        valid = (u >= -1e-9) & (u <= 1+1e-9) & (r >= 0)
        radii.extend(np.max(np.where(valid, r, -np.inf), axis=0))
    result = np.array(radii)
    if not np.isfinite(result).all():
        raise ValueError("Section is not a closed, origin-containing radial outline")
    return result


def periodicity(profile):
    residual = profile - profile.mean()
    spectrum = abs(np.fft.rfft(residual))
    search = spectrum[3:101]
    order = int(np.argmax(search) + 3)
    energy = spectrum[order]**2 / max(float(np.sum(spectrum[1:]**2)), 1e-30)
    peaks = (profile > np.roll(profile, 1)) & (profile >= np.roll(profile, -1))
    return {"dominant_order": order, "dominant_energy_fraction": float(energy),
            "radial_peak_to_peak_mm": float(np.ptp(profile)), "radial_max_mm": float(profile.max()),
            "raw_local_maxima": int(peaks.sum())}


def measure_sections(triangles_mm, count=27):
    lo, hi = triangles_mm[:, :, 0].min(), triangles_mm[:, :, 0].max()
    results = []
    # Avoid coincident triangle planes. Report all slices, including shoulders.
    for x in np.linspace(lo + .017*(hi-lo), hi - .023*(hi-lo), count):
        segments = section_segments(triangles_mm, x)
        if not len(segments):
            continue
        try:
            profile = radial_profile(segments)
            metrics = periodicity(profile)
            results.append({"x_mm": float(x), **metrics})
        except ValueError as exc:
            results.append({"x_mm": float(x), "error": str(exc)})
    evidence = [r for r in results if r.get("radial_peak_to_peak_mm", 0) > .10 and
                r.get("dominant_energy_fraction", 0) > .50]
    votes = Counter(r["dominant_order"] for r in evidence)
    winner, votes_count = votes.most_common(1)[0] if votes else (None, 0)
    return {"sections": results, "strong_periodicity_votes": dict(votes),
            "measured_periodic_order": winner if votes_count >= 3 else None,
            "supporting_sections": votes_count,
            "method": "radial outline sections; FFT orders searched from 3 to 100 without expected tooth number"}
