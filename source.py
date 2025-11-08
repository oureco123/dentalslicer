# -----------------------------------------------------------------------------
# Digital Cross Section Mapping Workflow
# Copyright (c) 2025 Majed Kutaini, RWTH Aachen University
#
# License: Research and Educational Use Only
# This software may be used, copied, modified, and distributed for
# academic and educational purposes with proper citation.
#
# Commercial use, in whole or in part, is strictly prohibited without
# prior written consent from the author.
#
# The full license is available in the LICENSE file in the root directory 
# of this source tree.
#
# Contact: kutainimajed@gmail.com
# -----------------------------------------------------------------------------

import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.ndimage import zoom
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from numba import njit, prange
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline
from sklearn.neighbors import NearestNeighbors
from vedo import Plotter, Mesh, Point, Plane
import time
import csv

# v ist der Modellname (Modell 1 = M1, usw)
# xx ist die Zahnnummerierung nach FDI
v = "M1"
xx = "11"

model_file = f'Zähne/{v}/{xx}/{xx}_umschlagpunkt.stl'
prep_file = f'Zähne/{v}/{xx}/{xx}_präpgrenze_d.stl'  # Path to the präpgrenze STL file
prep_k_file = f'Zähne/{v}/{xx}/{xx}_präpgrenze_k.stl' # konv. abformung

@njit(parallel=True)
def batch_line_intersections(line_starts, line_ends, contour_points):
    """
    Compute intersections between multiple scan lines and contour segments in parallel
    Returns intersection points and their corresponding line indices
    """
    intersections = []
    line_indices = []
    
    n_lines = len(line_starts)
    n_contour = len(contour_points) - 1
    
    for line_idx in prange(n_lines):
        x1, y1 = line_starts[line_idx]
        x2, y2 = line_ends[line_idx]
        
        for seg_idx in range(n_contour):
            x3, y3 = contour_points[seg_idx]
            x4, y4 = contour_points[seg_idx + 1]
            
            # Fast line intersection
            denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
            if abs(denom) > 1e-10:  # Not parallel
                ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
                ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
                
                if 0 <= ua <= 1 and 0 <= ub <= 1:
                    x = x1 + ua*(x2-x1)
                    y = y1 + ua*(y2-y1)
                    intersections.append((x, y))
                    line_indices.append(line_idx)
    
    return intersections, line_indices
@njit
def find_point_fast(slice_data, mode='bottom_right'):
    """Optimized point finding using numba JIT compilation"""
    points = np.where(slice_data > 0)
    if len(points[0]) == 0:
        return None
    
    rows, cols = points
    
    if mode == 'bottom_right':
        scores = cols - rows  # x - y for bottom-right
        best_idx = np.argmax(scores)
    elif mode == 'bottom_left':
        scores = -cols - rows  # -x - y for bottom-left
        best_idx = np.argmax(scores)
    elif mode == 'bottom':
        best_idx = np.argmin(rows)  # minimum Y
    elif mode == 'left':
        best_idx = np.argmax(rows)  # maximum Y
    else:
        best_idx = 0
    
    return (cols[best_idx], rows[best_idx])

@njit
def line_intersection_fast(x1, y1, x2, y2, x3, y3, x4, y4):
    """Fast line intersection using numba"""
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if abs(denom) < 1e-10:  # Lines are parallel
        return None, None
        
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua*(x2-x1)
        y = y1 + ua*(y2-y1)
        return x, y
    return None, None

@njit
def rasterize_line_fast(image, x1, y1, x2, y2):
    """Fast line rasterization using Bresenham's algorithm with numba"""
    height, width = image.shape
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    x_step = 1 if x1 < x2 else -1
    y_step = 1 if y1 < y2 else -1
    
    x, y = x1, y1
    
    if dx > dy:
        err = dx / 2
        while x != x2:
            if 0 <= x < width and 0 <= y < height:
                image[y, x] = 1.0
            err -= dy
            if err < 0:
                y += y_step
                err += dx
            x += x_step
    else:
        err = dy / 2
        while y != y2:
            if 0 <= x < width and 0 <= y < height:
                image[y, x] = 1.0
            err -= dx
            if err < 0:
                x += x_step
                err += dy
            y += y_step
    
    # Mark final point
    if 0 <= x2 < width and 0 <= y2 < height:
        image[y2, x2] = 1.0

def _bilinear(img, x, y):
    h, w = img.shape
    if x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return 0.0
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1, w-1), min(y0+1, h-1)
    dx, dy = x - x0, y - y0
    v00 = img[y0, x0]; v10 = img[y0, x1]; v01 = img[y1, x0]; v11 = img[y1, x1]
    return (v00*(1-dx)*(1-dy) + v10*dx*(1-dy) + v01*(1-dx)*dy + v11*dx*dy)

def _clamp_segment_to_image(p1, p2, h, w, eps=1e-6):
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    d = p2 - p1
    tmin, tmax = 0.0, 1.0
    for coord in range(2):
        p, dp = p1[coord], d[coord]
        lo, hi = 0.0, (w-eps if coord == 0 else h-eps)
        if abs(dp) < 1e-14:
            if p < lo or p > hi:
                return None
        else:
            t0, t1 = (lo - p)/dp, (hi - p)/dp
            t_enter, t_exit = min(t0, t1), max(t0, t1)
            tmin, tmax = max(tmin, t_enter), min(tmax, t_exit)
            if tmin > tmax:
                return None
    # pad a hair so we don't lose grazing intersections at the boundary
    pad = 1e-4
    return max(0.0, tmin - pad), min(1.0, tmax + pad)

def intersect_margin_with_slice(slice_data, p1, p2, prefer="last"):
    """
    Robust sub-pixel intersection of line segment (p1,p2) with the slice boundary.
    Works even when raster overlap is empty and across tiny angle changes.
    """
    field = (slice_data > 0).astype(np.float32)   # if you already have grayscale, use it instead

    h, w = field.shape
    rng = _clamp_segment_to_image(p1, p2, h, w)
    if rng is None:
        return None
    t0, t1 = rng

    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    d  = p2 - p1
    L  = np.hypot(d[0], d[1])
    if L < 1e-9:
        return None

    def find_crossings(step):
        n = max(2, int(np.ceil((t1 - t0) * L / step)))
        ts = np.linspace(t0, t1, n)
        vals = np.array([_bilinear(field, *(p1 + t*d)) for t in ts]) - 0.5
        crossings = []
        for i in range(1, len(ts)):
            a, b = ts[i-1], ts[i]
            va, vb = vals[i-1], vals[i]

            # relaxed crossing test: any interval that straddles zero or touches it
            if (va == 0.0) or (vb == 0.0) or (va < 0) != (vb < 0):
                # refine by bisection
                lo, hi, vlo, vhi = a, b, va, vb
                for _ in range(30):
                    m = 0.5*(lo + hi)
                    vm = _bilinear(field, *(p1 + m*d)) - 0.5
                    # treat |vm| < tol as a hit to avoid endless flip-flop
                    if abs(vm) < 1e-8:
                        lo = hi = m
                        break
                    if (vlo < 0) != (vm < 0):
                        hi, vhi = m, vm
                    else:
                        lo, vlo = m, vm
                crossings.append(0.5*(lo + hi))
        return crossings

    # 1) coarse pass, then 2) fine pass if needed
    crossings = find_crossings(step=0.4)
    if not crossings:
        crossings = find_crossings(step=0.1)
    if not crossings:
        return None

    t = crossings[-1] if prefer == "last" else crossings[0]
    pt = p1 + t*d
    return float(pt[0]), float(pt[1])



def intersect_margin_with_slice_advanced(slice_data, p1, p2, ref_point=None, min_dist_px=0, mode="first_after"):
    """
    Find sub-pixel intersections of the segment (p1->p2) with the binary slice.
    Selection rules controlled by `mode`:
      - "first_after": pick the FIRST intersection strictly after `ref_point` along the segment direction
                       (parametric t increasing from p1 to p2), ignoring any intersections whose
                       Euclidean distance to `ref_point` is < min_dist_px. This matches the requirement:
                       "first intersection to the right of präpgrenze_k_right, ignoring the tiny 3–5px hit".
      - "nearest_any": legacy mode; from all valid intersections (t between 0 and 1, inside segment),
                       ignore those within `min_dist_px` of `ref_point` (if provided) and pick the one with
                       smallest Euclidean distance to `ref_point`.
    Returns (x, y) or None if no valid intersection exists.
    """
    field = (slice_data > 0).astype(np.float32)

    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    d = p2 - p1
    seg_len = float(np.linalg.norm(d))
    if seg_len < 1e-6:
        return None
    u = d / seg_len  # unit direction

    # Sample the segment densely and detect threshold crossings (0 -> 1 or 1 -> 0)
    n_steps = max(2, int(seg_len * 2))  # 0.5px step
    ts = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    vals = []
    for t in ts:
        p = p1 + t * d
        x, y = p[0], p[1]
        ix, iy = int(np.floor(x)), int(np.floor(y))
        if 0 <= ix < field.shape[1]-1 and 0 <= iy < field.shape[0]-1:
            # bilinear
            dx, dy = x - ix, y - iy
            v00 = field[iy, ix]
            v10 = field[iy, ix+1]
            v01 = field[iy+1, ix]
            v11 = field[iy+1, ix+1]
            v0 = v00*(1-dx) + v10*dx
            v1 = v01*(1-dx) + v11*dx
            v = v0*(1-dy) + v1*dy
        else:
            v = 0.0
        vals.append(v)
    vals = np.array(vals, dtype=np.float32)

    # detect zero-crossings around 0.5
    crossings = []
    for i in range(len(ts)-1):
        if (vals[i] - 0.5) * (vals[i+1] - 0.5) < 0:  # sign change across 0.5
            t0, t1 = ts[i], ts[i+1]
            v0, v1 = vals[i], vals[i+1]
            if abs(v1 - v0) < 1e-6:
                t = 0.5 * (t0 + t1)
            else:
                t = t0 + (0.5 - v0) * (t1 - t0) / (v1 - v0)
            pt = p1 + t * d
            crossings.append((t, pt))

    if not crossings:
        return None

    # If ref_point is given, compute its parametric t on the segment
    tref = None
    ref = None
    if ref_point is not None:
        ref = np.array(ref_point, dtype=np.float32)
        # project onto the segment
        w = ref - p1
        tref = float(np.dot(w, d) / (seg_len * seg_len))

    def valid_entry(entry):
        t, pt = entry
        if not (0.0 <= t <= 1.0):
            return False
        if ref is not None:
            # must be strictly AFTER the reference along the segment
            if t <= tref:
                return False
            # and far enough from the reference to ignore tiny immediate intersections
            if np.linalg.norm(pt - ref) < float(min_dist_px):
                return False
        return True

    candidates = [e for e in crossings if valid_entry(e)]
    if not candidates:
        return None

    if mode == "first_after":
        # pick the smallest t after tref
        candidates.sort(key=lambda e: e[0])
        t, pt = candidates[0]
        return float(pt[0]), float(pt[1])
    elif mode == "nearest_any":
        if ref is None:
            # fall back to first in param order if no reference
            candidates.sort(key=lambda e: e[0])
            t, pt = candidates[0]
            return float(pt[0]), float(pt[1])
        # pick by Euclidean distance to ref
        t, pt = min(candidates, key=lambda e: np.linalg.norm(e[1]-ref))
        return float(pt[0]), float(pt[1])
    else:
        # default to first_after behavior
        candidates.sort(key=lambda e: e[0])
        t, pt = candidates[0]
        return float(pt[0]), float(pt[1])

    # Reference t and along-segment threshold
    ref = np.asarray(ref_point, float)
    denom = float(np.dot(d, d))
    t_ref = 0.0 if denom == 0.0 else float(np.dot(ref - p1, d) / denom)
    # translate the min_dist in px into parametric t distance
    min_dt = float(min_dist_px) / max(L, 1e-9)
    min_t = t_ref + min_dt

    # Filter: only crossings strictly after min_t
    after = [(t, p1 + t*d) for t in crossings if t > min_t]
    if not after:
        return None

    after.sort(key=lambda x: x[0])  # increasing in t
    choice = after[-2] if len(after) >= 2 else after[-1]
    pt = choice[1]
    return float(pt[0]), float(pt[1])

    # Compute reference t along the segment
    ref = np.asarray(ref_point, float)
    denom = np.dot(d, d)
    t_ref = 0.0 if denom == 0 else float(np.dot(ref - p1, d) / denom)

    # Build (t, point) for all crossings
    pts = [(t, p1 + t*d) for t in crossings]

    # Filter: only crossings *after* ref point and at least min_dist_px away from ref_point
    def ok(t, pt):
        if t <= t_ref:
            return False
        if min_dist_px > 0:
            if np.linalg.norm(pt - ref) < float(min_dist_px):
                return False
        return True

    after = [(t, pt) for (t, pt) in pts if ok(t, pt)]
    if not after:
        return None

    # Sort by t increasing, choose second-to-last if possible
    after.sort(key=lambda x: x[0])
    choice = after[-2] if len(after) >= 2 else after[-1]
    pt = choice[1]
    return float(pt[0]), float(pt[1])

# 2. Optimized mesh intersection class
class OptimizedMeshProcessor:
    def __init__(self, shared_bounds, slice_resolution):
        self.shared_bounds = shared_bounds
        self.slice_resolution = slice_resolution
        self.intersection_cache = {}  # Cache mesh intersections
        self.bounds_cache = {}        # Cache transformed bounds
        
        # Pre-calculate transformation matrices for common angles
        self.precomputed_transforms = {}
        
        # Calculate bounds once
        self.bounds_2d = np.array([
            [shared_bounds[0][1], shared_bounds[0][2]],  # Y_min, Z_min
            [shared_bounds[1][1], shared_bounds[1][2]]   # Y_max, Z_max
        ])
        
        # Pre-calculate scaling factors
        self.y_range = max(self.bounds_2d[1][0] - self.bounds_2d[0][0], 1e-6)
        self.z_range = max(self.bounds_2d[1][1] - self.bounds_2d[0][1], 1e-6)
        self.y_scale = (slice_resolution - 1) / self.y_range
        self.z_scale = (slice_resolution - 1) / self.z_range
    
    def get_plane_intersection_cached(self, mesh, angle, plane_origin, plane_normal):
        cache_key = (id(mesh), angle, tuple(plane_origin), tuple(plane_normal))
        if cache_key not in self.intersection_cache:
            result = self.get_plane_intersection(mesh, plane_origin, plane_normal)
            self.intersection_cache[cache_key] = result
        return self.intersection_cache[cache_key]
    
    def get_rotation_matrix(self, angle_degrees):
        """Cache rotation matrices for better performance"""
        if angle_degrees not in self.precomputed_transforms:
            angle_rad = np.deg2rad(angle_degrees)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            self.precomputed_transforms[angle_degrees] = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
        return self.precomputed_transforms[angle_degrees]
    
    def rasterize_lines_optimized(self, lines_list):
        """Optimized line rasterization"""
        image = np.zeros((self.slice_resolution, self.slice_resolution), dtype=np.uint8)
        
        if not lines_list:
            return np.zeros((self.slice_resolution, self.slice_resolution), dtype=np.float32)
        
        for lines in lines_list:
            if len(lines) < 2:
                continue
            
            # Convert to image coordinates
            y_coords = ((lines[:, 0] - self.bounds_2d[0][0]) * self.y_scale).astype(np.int32)
            z_coords = ((lines[:, 1] - self.bounds_2d[0][1]) * self.z_scale).astype(np.int32)
            
            # Clamp coordinates
            y_coords = np.clip(y_coords, 0, self.slice_resolution - 1)
            z_coords = np.clip(z_coords, 0, self.slice_resolution - 1)
            
            # Draw lines using optimized function
            for i in range(len(y_coords) - 1):
                rasterize_line_fast(image, y_coords[i], z_coords[i], 
                                  y_coords[i+1], z_coords[i+1])
        
        return image

# Configure TensorFlow for better resource usage
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(42)


# Custom ICP implementation (unchanged)
def best_fit_transform(A, B):
    assert A.shape == B.shape
    m = A.shape[1]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.reshape(-1,1) - np.dot(R, centroid_A.reshape(-1,1))
    T = np.eye(m+1)
    T[:m, :m] = R
    T[:m, -1] = t.ravel()
    return T

def nearest_neighbor(src, dst):
    assert src.shape == dst.shape
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def iterative_closest_point(A, B, max_iterations=20, tolerance=0.001):
    assert A.shape == B.shape
    m = A.shape[1]
    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    prev_error = 0
    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        T = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        src = np.dot(T, src)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    T = best_fit_transform(A, src[:m,:].T)
    final_error = prev_error
    rot = T[0:-1,0:-1]
    t = T[:-1,-1]
    finalA = np.dot(rot, A.T).T + t
    return T, finalA, final_error, i

def align_meshes_interactively(prep_mesh, prep_k_mesh, tooth_id):
    """Align prep_k_mesh to prep_mesh using ICP with interactive point selection."""
    # Convert trimesh objects to vedo meshes
    prep_mesh_vedo = Mesh([prep_mesh.vertices, prep_mesh.faces]).color("red").legend("Target Mesh")
    prep_k_mesh_vedo = Mesh([prep_k_mesh.vertices, prep_k_mesh.faces]).color("blue").legend("Source Mesh")

    # Create a vedo Plotter for interactive point selection
    plt = Plotter(title="Left: Red (Dig), Right: Blue (Konv)")

    # Lists to store selected points
    source_points = []
    target_points = []
    drawn_points = []

    # Callback function to select points on the source mesh
    def select_source_point(evt):
        if evt.actor and evt.actor == prep_k_mesh_vedo:  # Ensure the click is on the blue model
            point = evt.picked3d
            source_points.append(point)
            print(f"Selected source point: {point}")
            
            # Draw a blue dot at the selected point
            dot = Point(point, c="red", r=10)
            drawn_points.append(dot)
            plt.add(dot)

    # Callback function to select points on the target mesh
    def select_target_point(evt):
        if evt.actor and evt.actor == prep_mesh_vedo:  # Ensure the click is on the red model
            point = evt.picked3d
            target_points.append(point)
            print(f"Selected target point: {point}")
            
            # Draw a red dot at the selected point
            dot = Point(point, c="blue", r=10)
            drawn_points.append(dot)
            plt.add(dot)

    def delete_last_point(evt):
        if evt.keypress == "Backspace":  # Only react to the Backspace key
            if drawn_points:
                last_point = drawn_points.pop()
                plt.remove(last_point)
                
                # Remove the last point from the corresponding list
                if source_points and np.array_equal(source_points[-1], last_point.pos()):
                    source_points.pop()
                elif target_points and np.array_equal(target_points[-1], last_point.pos()):
                    target_points.pop()
                print("Last point deleted.")
                plt.render() 

    # Add meshes to the plotter
    plt.add(prep_mesh_vedo)
    plt.add(prep_k_mesh_vedo)

    # Set up callbacks for point selection
    plt.add_callback('LeftButtonPress', select_source_point)
    plt.add_callback('RightButtonPress', select_target_point)
    plt.add_callback('KeyPress', delete_last_point)  # Bind delete key to delete_last_point

    # Show the interactive viewer
    plt.show()

    # Convert selected points to numpy arrays
    source_points = np.array(source_points)
    target_points = np.array(target_points)

    # Ensure at least 3 points are selected for alignment
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError("Please select at least 3 corresponding points on both meshes.")

    # Save the selected points to a file
    icp_points = {
        "source_points": source_points,
        "target_points": target_points,
        "is_flipped": False
    }
    icp_filename = f"Zähne/{v}/{xx}/icp_points_{tooth_id}.npy"
    np.save(icp_filename, icp_points)
    print(f"ICP points saved to {icp_filename}")

    # Compute initial transformation using manually selected points
    initial_T = best_fit_transform(source_points, target_points)

    # Apply the initial transformation to the source mesh
    prep_k_mesh.apply_transform(initial_T)

    # Sample points from both meshes for ICP
    target_points_icp = prep_mesh.sample(5000)  # Points from präpgrenze (target)
    source_points_icp = prep_k_mesh.sample(5000)  # Points from präpgrenze_k (source)

    # Run custom ICP to align präpgrenze_k to präpgrenze
    T, finalA, final_error, iterations = iterative_closest_point(
        source_points_icp, 
        target_points_icp,
        max_iterations=500,  # Increase iterations for better alignment
        tolerance=1e-6       # Smaller tolerance for finer alignment
    )

    # Apply the ICP transformation to the entire präpgrenze_k mesh
    prep_k_mesh.apply_transform(T)

    # Assign darker colors to the meshes
    prep_mesh_color = [128, 0, 0, 255]  # Dark red for präpgrenze (RGBA)
    prep_k_mesh_color = [0, 0, 128, 255]  # Dark blue for präpgrenze_k (RGBA)

    # Apply colors to the meshes
    prep_mesh.visual.face_colors = prep_mesh_color
    prep_k_mesh.visual.face_colors = prep_k_mesh_color

    # Convert trimesh objects to vedo meshes
    prep_mesh_vedo = Mesh([prep_mesh.vertices, prep_mesh.faces]).color("red").legend("Target Mesh")
    prep_k_mesh_vedo = Mesh([prep_k_mesh.vertices, prep_k_mesh.faces]).color("blue").legend("Source Mesh")

    # Create a vedo Plotter
    plt = Plotter(title="Aligned Meshes")
    plt.add(prep_mesh_vedo)
    plt.add(prep_k_mesh_vedo)
    plt.show()  # This will block until the window is closed

    # Print alignment metrics
    print(f"Final alignment error: {final_error:.6f}")
    print(f"Number of iterations: {iterations}")
    return prep_k_mesh

def adjust_mesh_to_bounds(mesh, bounds):
    """Translate and scale the mesh to fit within the specified bounds."""
    # Calculate the current bounds of the mesh
    mesh_bounds = mesh.bounds
    
    # Calculate the required translation and scaling
    translation = (bounds[0] - mesh_bounds[0])  # Translate to align min bounds
    scaling = (bounds[1] - bounds[0]) / (mesh_bounds[1] - mesh_bounds[0])  # Scale to fit within bounds
    
    # Apply translation and scaling to the mesh
    mesh.apply_translation(translation)
    mesh.apply_scale(scaling)
    
    return mesh
    
class InteractiveRotationViewer:
    def __init__(self, model_file, prep_file, prep_k_file, num_angles=360, slice_resolution=500, batch_size=20):
        self.model_file = model_file
        self.prep_file = prep_file
        self.prep_k_file = prep_k_file
        self.num_angles = num_angles
        self.slice_resolution = slice_resolution
        self.batch_size = min(60, cpu_count() * 6)  # Auto-adjusts to your core count
        self.marked_point = None
        self.model = None
        self.model_weights_file = "tooth_segmentation_model.weights.h5"
        self.correction_data = []
        self.last_retrain_size = 0  # Track training data size
        self.angle_corrections = {}  # Track corrections per angle
        self.current_correction_id = 0  # Unique ID for corrections
        self.marked_umschlagpunkt = None
        self.mesh_processor = None
        self.is_flipped = False
        
        # Initialize model and data
        self.initialize_model()
        self.load_and_process_meshes()
        self.precompute_slice_data()


    def initialize_model(self):
        """Improved model for spatial-angular learning"""
        input_shape = (self.slice_resolution, self.slice_resolution, 3)
        
        inputs = layers.Input(shape=input_shape)
        
        # Angular pathway
        angle_features = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(inputs)
        angle_features = layers.MaxPooling2D((2, 2))(angle_features)
        
        # Spatial pathway
        spatial_features = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        spatial_features = layers.MaxPooling2D((2, 2))(spatial_features)
        
        # Ensure both pathways have the same spatial dimensions
        # If necessary, apply padding or cropping
        if spatial_features.shape[1] != angle_features.shape[1] or spatial_features.shape[2] != angle_features.shape[2]:
            # Apply padding to match dimensions
            spatial_features = layers.ZeroPadding2D(((0, 1), (0, 1)))(spatial_features)
        
        # Merge pathways
        merged = layers.Concatenate()([spatial_features, angle_features])
        merged = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merged)
        merged = layers.GlobalAveragePooling2D()(merged)
        
        outputs = layers.Dense(2, activation='sigmoid')(merged)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mse')

    def load_and_process_meshes(self):
        #print("Loading and processing meshes...")
        
        # Load meshes
        self.mesh = trimesh.load(self.model_file)  # umschlagpunkt
        self.prep_mesh = trimesh.load(self.prep_file)  # präpgrenze
        self.prep_k_mesh = trimesh.load(self.prep_k_file)  # präpgrenze_konventionell

        # Print mesh information before translation
        """
        print("\nMesh Information (Before Translation):")
        print(f"Umschlagpunkt Mesh Bounds: {self.mesh.bounds}")
        print(f"Präpgrenze Mesh Bounds: {self.prep_mesh.bounds}")
        print(f"Präpgrenze_k Mesh Bounds: {self.prep_k_mesh.bounds}")
        print(f"Umschlagpunkt Mesh Center: {self.mesh.centroid}")
        print(f"Präpgrenze Mesh Center: {self.prep_mesh.centroid}")
        print(f"Präpgrenze_k Mesh Center: {self.prep_k_mesh.centroid}")
        """

        # Generate tooth ID from v and xx
        tooth_id = f"{v}{xx}"
        icp_filename = f"Zähne/{v}/{xx}/icp_points_{tooth_id}.npy"

        # Check if ICP points are saved
        if os.path.exists(icp_filename):
            #print(f"Loading ICP points from {icp_filename}...")
            icp_points = np.load(icp_filename, allow_pickle=True).item()
            source_points = icp_points["source_points"]
            target_points = icp_points["target_points"]

            self.is_flipped = icp_points.get("is_flipped", False)

            # Compute initial transformation using saved points
            initial_T = best_fit_transform(source_points, target_points)
            self.prep_k_mesh.apply_transform(initial_T)

            # Sample points from both meshes for ICP
            target_points_icp = self.prep_mesh.sample(5000)  # Points from präpgrenze (target)
            source_points_icp = self.prep_k_mesh.sample(5000)  # Points from präpgrenze_k (source)

            # Run custom ICP to align präpgrenze_k to präpgrenze
            T, finalA, final_error, iterations = iterative_closest_point(
                source_points_icp, 
                target_points_icp,
                max_iterations=500,  # Increase iterations for better alignment
                tolerance=1e-6       # Smaller tolerance for finer alignment
            )

            # Apply the ICP transformation to the entire präpgrenze_k mesh
            self.prep_k_mesh.apply_transform(T)
        else:
            # If no saved points, run interactive ICP
            print(f"No saved ICP points found for tooth {tooth_id}. Running interactive ICP...")
            self.prep_k_mesh = align_meshes_interactively(self.prep_mesh, self.prep_k_mesh, tooth_id)
 
        # Print mesh information after translation
        """
        print("\nMesh Information (After Translation):")
        print(f"Umschlagpunkt Mesh Bounds: {self.mesh.bounds}")
        print(f"Präpgrenze Mesh Bounds: {self.prep_mesh.bounds}")
        print(f"Präpgrenze_k Mesh Bounds: {self.prep_k_mesh.bounds}")
        print(f"Umschlagpunkt Mesh Center: {self.mesh.centroid}")
        print(f"Präpgrenze Mesh Center: {self.prep_mesh.centroid}")
        print(f"Präpgrenze_k Mesh Center: {self.prep_k_mesh.centroid}")
        """

        # Calculate the shared bounding box that encompasses all meshes
        self.shared_bounds = np.array([
            np.minimum(
                np.minimum(self.mesh.bounds[0], self.prep_mesh.bounds[0]),
                self.prep_k_mesh.bounds[0]
            ),  # Min bounds
            np.maximum(
                np.maximum(self.mesh.bounds[1], self.prep_mesh.bounds[1]),
                self.prep_k_mesh.bounds[1]
            )   # Max bounds
        ])

        # Print shared bounding box information
        #print("\nShared Bounding Box:")
        #print(f"Min Bounds: {self.shared_bounds[0]}")
        #print(f"Max Bounds: {self.shared_bounds[1]}")

        # === OPTIMIZATION 5: Initialize optimized processor ===
        self.mesh_processor = OptimizedMeshProcessor(self.shared_bounds, self.slice_resolution)
        # Calculate mm per pixel for distance calculations (matching original calculation)
        bounds_size = self.shared_bounds[1] - self.shared_bounds[0]
        # Use the same calculation as the original voxel approach
        max_dimension = max(bounds_size)
        self.mm_per_pixel = max_dimension / self.slice_resolution
        
        #print(f"MM per pixel: {self.mm_per_pixel}")
        #print(f"Max dimension: {max_dimension}")
        #print(f"Slice resolution: {self.slice_resolution}")

    def rotate_mesh_around_point(self, mesh, angle_degrees, center_point):
        """Rotate mesh around a specific point (matching voxel rotation behavior)"""
        angle_rad = np.deg2rad(angle_degrees)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle_rad, [0, 0, 1], center_point  # Rotate around Z-axis at shared center
        )
        rotated_mesh = mesh.copy()
        rotated_mesh.apply_transform(rotation_matrix)
        return rotated_mesh
        
    def get_plane_intersection(self, mesh, plane_origin, plane_normal):
        """Get intersection of mesh with plane"""
        try:
            # Use trimesh's built-in plane intersection
            intersection = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            
            if intersection is None:
                return []
                
            # Convert to 2D coordinates in the plane
            # For X-plane (normal=[1,0,0]), we want Y-Z coordinates
            if hasattr(intersection, 'entities'):
                lines = []
                for entity in intersection.entities:
                    if hasattr(entity, 'points'):
                        # Get 3D points and project to 2D
                        points_3d = intersection.vertices[entity.points]
                        # For X-plane, use Y and Z coordinates
                        points_2d = points_3d[:, [1, 2]]  # Y, Z
                        lines.append(points_2d)
                return lines
            else:
                # Single path case
                if len(intersection.vertices) > 0:
                    points_2d = intersection.vertices[:, [1, 2]]  # Y, Z
                    return [points_2d]
                    
        except Exception as e:
            print(f"Intersection error: {e}")
            
        return []
        
    def rasterize_lines_to_image(self, lines_list, bounds_2d, resolution):
        """Convert line segments to rasterized 2D image (matching voxel slice orientation)"""
        image = np.zeros((resolution, resolution), dtype=np.float32)
        
        if not lines_list:
            return image
            
        # Calculate scaling from world coordinates to image coordinates
        y_min, z_min = bounds_2d[0]
        y_max, z_max = bounds_2d[1]
        
        # Ensure we don't divide by zero
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        if y_range == 0:
            y_range = 1e-6
        if z_range == 0:
            z_range = 1e-6
            
        y_scale = (resolution - 1) / y_range
        z_scale = (resolution - 1) / z_range
        
        for lines in lines_list:
            if len(lines) < 2:
                continue
                
            # Convert world coordinates to image coordinates
            # Note: Z becomes the row (image Y), Y becomes the column (image X)
            y_coords = ((lines[:, 0] - y_min) * y_scale).astype(int)  # Y -> image X
            z_coords = ((lines[:, 1] - z_min) * z_scale).astype(int)  # Z -> image Y
            
            # Clamp coordinates to image bounds
            y_coords = np.clip(y_coords, 0, resolution - 1)
            z_coords = np.clip(z_coords, 0, resolution - 1)
            
            # Draw line segments - Note: image[row, col] = image[z, y]
            for i in range(len(y_coords) - 1):
                self.draw_line(image, 
                             (z_coords[i], y_coords[i]),    # (row, col) = (z, y)
                             (z_coords[i+1], y_coords[i+1]))
                             
            # Also mark individual points
            valid_indices = (z_coords >= 0) & (z_coords < resolution) & (y_coords >= 0) & (y_coords < resolution)
            if np.any(valid_indices):
                image[z_coords[valid_indices], y_coords[valid_indices]] = 1.0
            
        return image
        
    def draw_line(self, image, p1, p2):
        """Draw line between two points using Bresenham's algorithm"""
        row1, col1 = p1  # Note: using row/col terminology for clarity
        row2, col2 = p2
        
        # Bresenham's line algorithm
        drow = abs(row2 - row1)
        dcol = abs(col2 - col1)
        
        row_step = 1 if row1 < row2 else -1
        vop_step = 1 if col1 < col2 else -1
        
        if dcol > drow:
            err = dcol / 2
            row = row1
            for col in range(col1, col2 + vop_step, vop_step):
                if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
                    image[row, col] = 1.0
                err -= drow
                if err < 0:
                    row += row_step
                    err += dcol
        else:
            err = drow / 2
            col = col1
            for row in range(row1, row2 + row_step, row_step):
                if 0 <= row < image.shape[0] and 0 <= col < image.shape[1]:
                    image[row, col] = 1.0
                err -= dcol
                if err < 0:
                    col += vop_step
                    err += drow

    def get_slice_at_angle(self, angle_degrees):
        """Get 2D slice at given rotation angle"""
        # Rotate all meshes around their shared centroid (like in original)
        shared_centroid = (self.shared_bounds[0] + self.shared_bounds[1]) / 2
        
        # === OPTIMIZATION 7: Use cached rotation matrices ===
        rotation_matrix = self.mesh_processor.get_rotation_matrix(angle_degrees)
        
        tooth_rotated = self.rotate_mesh_around_point(self.mesh, angle_degrees, shared_centroid)
        prep_rotated = self.rotate_mesh_around_point(self.prep_mesh, angle_degrees, shared_centroid)
        prep_k_rotated = self.rotate_mesh_around_point(self.prep_k_mesh, angle_degrees, shared_centroid)
        
        # Define cutting plane (X = center, like in voxel approach)
        center_x = shared_centroid[0]
        plane_origin = np.array([center_x, 0, 0])
        plane_normal = np.array([1, 0, 0])
        
        # Get intersections
        tooth_lines = self.get_plane_intersection(tooth_rotated, plane_origin, plane_normal)
        prep_lines = self.get_plane_intersection(prep_rotated, plane_origin, plane_normal)
        prep_k_lines = self.get_plane_intersection(prep_k_rotated, plane_origin, plane_normal)
        
        
        # Rasterize to images
        tooth_image = self.mesh_processor.rasterize_lines_optimized(tooth_lines)
        prep_image = self.mesh_processor.rasterize_lines_optimized(prep_lines)
        prep_k_image = self.mesh_processor.rasterize_lines_optimized(prep_k_lines)
        
        return tooth_image, prep_image, prep_k_image

    def precompute_slice_data(self):
        """Precompute all slice data for all angles"""
        #print("Precomputing slice data for all angles...")
        self.angles = np.linspace(0, 360, self.num_angles, endpoint=False)
        
        self.slice_data_cache = [None] * self.num_angles
        self.prep_slice_data_cache = [None] * self.num_angles
        self.prep_k_slice_data_cache = [None] * self.num_angles
        
        for i, angle in tqdm(enumerate(self.angles), desc="Computing slices"):
            tooth_img, prep_img, prep_k_img = self.get_slice_at_angle(angle)
            self.slice_data_cache[i] = tooth_img
            self.prep_slice_data_cache[i] = prep_img
            self.prep_k_slice_data_cache[i] = prep_k_img

    def flip_model_180(self, event):
        """Flip the display by 180 degrees and save state"""
        #print("Flipping view by 180 degrees...")
        self.is_flipped = not self.is_flipped
        
        # Save the flip state to ICP file
        tooth_id = f"{v}{xx}"
        icp_filename = f"Zähne/{v}/{xx}/icp_points_{tooth_id}.npy"
        if os.path.exists(icp_filename):
            icp_points = np.load(icp_filename, allow_pickle=True).item()
            icp_points["is_flipped"] = self.is_flipped
            np.save(icp_filename, icp_points)
        
        # Update the display
        current_angle_idx = int(self.slider.val)
        self.plot_slice(current_angle_idx)
        
        print(f"View flipped {'(reversed)' if self.is_flipped else '(normal)'}")

    def find_nearest_point_on_graph(self, x, y, slice_data):
        graph_points = np.column_stack(np.where(slice_data > 0))
        if len(graph_points) == 0:
            return None
        distances = np.sqrt((graph_points[:, 1] - x) ** 2 + (graph_points[:, 0] - y) ** 2)
        nearest_index = np.argmin(distances)
        nearest_point = graph_points[nearest_index]
        return nearest_point[1], nearest_point[0]

    
    # === OPTIMIZATION 12: Use optimized point finding ===
    def find_bottom_right(self, slice_data):
        return find_point_fast(slice_data, 'bottom_right')
    
    def find_bottom_left(self, slice_data):
        return find_point_fast(slice_data, 'bottom_left')
    
    def find_bottom(self, slice_data):
        return find_point_fast(slice_data, 'bottom')
    
    def find_left(self, slice_data):
        return find_point_fast(slice_data, 'left')
    
    
    def calculate_angle(a, b, c):
        """Calculate the angle between points a-b-c in degrees."""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def calculate_parallel_line_distance(self, line1_points, line2_points):
        """
        Calculate perpendicular distance between two parallel lines
        line1_points and line2_points are tuples of (start_point, end_point)
        """
        line1_start, line1_end = line1_points
        line2_start, line2_end = line2_points
        
        # Get direction vector of first line
        direction = np.array(line1_end) - np.array(line1_start)
        direction = direction / np.linalg.norm(direction)
        
        # Get perpendicular direction
        perp_direction = np.array([-direction[1], direction[0]])
        
        # Vector from line1 to line2
        to_line2 = np.array(line2_start) - np.array(line1_start)
        
        # Perpendicular distance
        distance_px = abs(np.dot(to_line2, perp_direction))
        distance_mm = distance_px * self.mm_per_pixel
        
        return distance_mm
    

    def calculate_mep_and_detection_points(self, slice_data, prep_slice_data, prep_k_slice_data):
        """
        Unified method to calculate mep, sulcusboden, and col points.
        Returns: (final_mep, current_sulcusboden, current_vop, extended_line_points)
        """
        # Find key points using optimized methods
        präpgrenze_left = self.find_bottom_left(prep_slice_data)
        präpgrenze_right = self.find_bottom_right(prep_slice_data)
        
        final_mep = None
        current_sulcusboden = None
        current_vop = None
        extended_line_points = None
        sulcus_distance_mm = None
        vop_distance_mm = None
        sulcustiefe = None
        
        # Calculate extended line and detection points
        if präpgrenze_left and präpgrenze_right:
            # Convert points to numpy arrays for vector math
            left = np.array(präpgrenze_left)
            right = np.array(präpgrenze_right)
            
            # Calculate direction vector
            direction = right - left
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            
            # Extended line for mep calculation
            extended_left = left - direction * 90
            extended_right = right + direction * 250
            extended_line_points = (extended_left, extended_right)
            
            # Lines for sulcus and col detection
            new_left = left + direction * 650
            new_right = right + direction * 150
            vop_left = right + direction * 40
            vop_right = right + direction * 250
            
            # Get contour points
            contour_points = np.column_stack(np.where(slice_data > 0))
            if len(contour_points) > 0:
                contour_points_xy = contour_points[:, [1, 0]]  # Convert to (x,y)
                perp_direction = np.array([-direction[1], direction[0]])
                # Detection algorithm
                step_size = 1
                max_steps = 500
                # === PRE-FILTER CONTOUR SEGMENTS ===
                # Only check segments that are in the scanning area
                sulcus_segments = self.get_relevant_segments(contour_points_xy, new_left, new_right, perp_direction, max_steps, negative=True)
                vop_segments = self.get_relevant_segments(contour_points_xy, vop_left, vop_right, perp_direction, max_steps, negative=False)
                
                for step in range(0, max_steps):
                    # === Sulcusboden Detection ===
                    reverse_step = max_steps - 1 - step
                    sulcus_offset = -perp_direction * reverse_step * step_size
                    shifted_left = new_left + sulcus_offset
                    shifted_right = new_right + sulcus_offset
                    
                    for i in range(len(sulcus_segments)-1):
                        p1 = sulcus_segments[i]
                        p2 = sulcus_segments[i+1]
                        sulcus_intersect = self.line_intersection((shifted_left, shifted_right), (p1, p2))
                        if sulcus_intersect:
                            current_sulcusboden = sulcus_intersect
                            if extended_line_points:
                                sulcus_distance_mm = self.calculate_parallel_line_distance(
                                    extended_line_points, 
                                    (shifted_left, shifted_right)
                                )
                               #print(f"Sulcusboden detection distance: {sulcus_distance_mm:.2f}mm")
                            break
                    
                    if current_sulcusboden:
                        break

                for step in range(0, max_steps):
                    # === Col Detection ===
                    reverse_step = max_steps - 1 - step
                    vop_offset = perp_direction * reverse_step * step_size
                    shifted_vop_left = vop_left + vop_offset
                    shifted_vop_right = vop_right + vop_offset
                    
                    for i in range(len(vop_segments)-1):
                        p1 = vop_segments[i]
                        p2 = vop_segments[i+1]
                        if np.linalg.norm(p2 - p1) < 2:  # Only adjacent points
                            vop_intersect = self.line_intersection((shifted_vop_left, shifted_vop_right), (p1, p2))
                            if vop_intersect:
                                dist_to_p1 = np.linalg.norm(np.array(vop_intersect) - np.array(p1))
                                dist_to_p2 = np.linalg.norm(np.array(vop_intersect) - np.array(p2))
                                if dist_to_p1 < 5 and dist_to_p2 < 5:
                                    current_vop = vop_intersect
                                    # Calculate distance from extended line to col detection line
                                    if extended_line_points:
                                        vop_distance_mm = self.calculate_parallel_line_distance(
                                            extended_line_points, 
                                            (shifted_vop_left, shifted_vop_right)
                                        )
                                        #print(f"Col detection distance: {vop_distance_mm:.2f}mm")
                                    break
                    
                    if current_vop:
                        break
        
        # === mep Calculation (robust) ===
        if extended_line_points is not None and slice_data is not None and slice_data.any():
            p1, p2 = extended_line_points          # keep as floats; do NOT round here
            mep = intersect_margin_with_slice_advanced(slice_data, p1, p2, ref_point=präpgrenze_right, min_dist_px=20, mode="first_after")
            if mep is not None:
                final_mep = mep
            else:
                final_mep = None

        
        if current_sulcusboden and current_vop:
            sulcustiefe = self.calculate_parallel_line_distance(
                                            (shifted_left, shifted_right), 
                                            (shifted_vop_left, shifted_vop_right)
                                        )
        return final_mep, current_sulcusboden, current_vop, extended_line_points, (shifted_left, shifted_right), (shifted_vop_left, shifted_vop_right), sulcus_distance_mm, vop_distance_mm, sulcustiefe


    def get_relevant_segments(self, contour_points_xy, base_left, base_right, perp_direction, max_steps, negative=True):
        """
        Filter contour segments to only those in the scanning area
        """
        if len(contour_points_xy) == 0:
            return contour_points_xy
        
        # Calculate bounding box of scanning area
        max_offset = max_steps
        direction = -perp_direction if negative else perp_direction
        
        # Four corners of scanning area
        corners = [
            base_left,
            base_right,
            base_left + direction * max_offset,
            base_right + direction * max_offset
        ]
        
        # Get bounding box with margin
        margin = 20
        min_x = min(corner[0] for corner in corners) - margin
        max_x = max(corner[0] for corner in corners) + margin
        min_y = min(corner[1] for corner in corners) - margin
        max_y = max(corner[1] for corner in corners) + margin
        
        # Filter points within bounding box
        relevant_points = []
        for point in contour_points_xy:
            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                relevant_points.append(point)
        
        return np.array(relevant_points) if relevant_points else np.array([]) 
    def plot_slice(self, angle_idx):
        angle = self.angles[angle_idx]
        
        # Get cached slice data
        slice_data_resized = self.slice_data_cache[angle_idx]
        prep_slice_data_resized = self.prep_slice_data_cache[angle_idx]
        prep_k_slice_data_resized = self.prep_k_slice_data_cache[angle_idx]

        if self.is_flipped:
            slice_data_resized = np.fliplr(np.flipud(slice_data_resized))
            prep_slice_data_resized = np.fliplr(np.flipud(prep_slice_data_resized))
            prep_k_slice_data_resized = np.fliplr(np.flipud(prep_k_slice_data_resized))

        self.ax.clear()
        from matplotlib.colors import ListedColormap
        # Base: inverted grayscale (0=white background, 1=black lines)
        base_cmap = 'gray_r'
        # Overlays: transparent for 0, solid color for 1 to avoid purple tints
        green_overlay = ListedColormap([(0, 0, 0, 0), (0, 0.8, 0, 1)])
        blue_overlay = ListedColormap([(0, 0, 0, 0), (0, 0.4, 1, 1)])

        self.ax.imshow(slice_data_resized, cmap=base_cmap, origin='lower', interpolation='none', vmin=0, vmax=1)
        self.ax.imshow(prep_slice_data_resized, cmap=green_overlay, origin='lower', interpolation='none', vmin=0, vmax=1)
        self.ax.imshow(prep_k_slice_data_resized, cmap=blue_overlay, origin='lower', interpolation='none', vmin=0, vmax=1)
        self.ax.set_title(f'Direct Mesh Intersection - Rotation angle: {angle:.2f}° around Z-axis, Slice along X-axis')

        # === Use unified calculation method ===
        final_mep, current_sulcusboden, current_vop, extended_line_points, sulcus_line, vop_line, sulcus_distance_mm, vop_distance_mm, sulcustiefe = \
            self.calculate_mep_and_detection_points(slice_data_resized, prep_slice_data_resized, prep_k_slice_data_resized)

        präpgrenze_left = self.find_bottom_left(prep_slice_data_resized)
        präpgrenze_right = self.find_bottom_right(prep_slice_data_resized)
        präpgrenze_k_right = self.find_bottom_right(prep_k_slice_data_resized)

        if präpgrenze_left:
            self.ax.plot(präpgrenze_left[0], präpgrenze_left[1], 'yo', markersize=8, label=f'Präpgrenze top ({präpgrenze_left[0]}, {präpgrenze_left[1]})')
        if präpgrenze_right:
            self.ax.plot(präpgrenze_right[0], präpgrenze_right[1], 'yo', markersize=8, label=f'Präpgrenze {präpgrenze_right[0]}, {präpgrenze_right[1]}')
        if präpgrenze_k_right:
            self.ax.plot(präpgrenze_k_right[0], präpgrenze_k_right[1], 'co', markersize=8, label=f'Präpgrenze_k {präpgrenze_k_right[0]}, {präpgrenze_k_right[1]}')
        
        if sulcus_line:
            self.ax.plot([sulcus_line[0][0], sulcus_line[1][0]],
                        [sulcus_line[0][1], sulcus_line[1][1]],
                        'b--', linewidth=1, alpha=0.5, label="Sulcusboden Detection Line")
        
        if vop_line:
            self.ax.plot([vop_line[0][0], vop_line[1][0]],
                        [vop_line[0][1], vop_line[1][1]],
                        'y--', linewidth=1, alpha=0.5, label="VOP Detection Line")
            
        if extended_line_points:
            extended_left, extended_right = extended_line_points
            self.ax.plot([extended_left[0], extended_right[0]],
                        [extended_left[1], extended_right[1]],
                        'k-', linewidth=1, alpha=0.7, label="Margin Extrapolation Line")
            
        if current_sulcusboden:
            self.ax.plot(current_sulcusboden[0], current_sulcusboden[1], 'bo', 
                        markersize=8, alpha=0.7, 
                        label=f'Sulcusboden ({current_sulcusboden[0]:.0f}, {current_sulcusboden[1]:.0f}); {sulcus_distance_mm} mm')

        if current_vop:
            self.ax.plot(current_vop[0], current_vop[1], 'yo', 
                        markersize=8, alpha=0.7, 
                        label=f'Vertex of the papilla ({current_vop[0]:.0f}, {current_vop[1]:.0f}); {vop_distance_mm} mm')

        # Plot mep (only if we found an intersection and distance <= 5mm)
        if final_mep and präpgrenze_right:
            distance_px = np.sqrt((final_mep[0] - präpgrenze_right[0])**2 +
                                (final_mep[1] - präpgrenze_right[1])**2)
            distance_mm = distance_px * self.mm_per_pixel
            
            if 0.1 <= distance_mm <= 3:    
                self.ax.plot(final_mep[0], final_mep[1], 'mo', 
                            markersize=8, alpha=0.7, 
                            label=f'Margin Extrapolation Point ({final_mep[0]:.0f}, {final_mep[1]:.0f}) - {distance_mm:.2f}mm')
                
                # Draw connecting line
                self.ax.plot([final_mep[0], präpgrenze_right[0]],
                            [final_mep[1], präpgrenze_right[1]],
                            'm-', linewidth=1, alpha=0.5)

        # ====== Update Legend ======
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.draw()

    # Add this new helper method to the class
    def line_intersection(self, line1, line2):
        """Optimized line intersection"""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        x, y = line_intersection_fast(x1, y1, x2, y2, x3, y3, x4, y4)
        return (x, y) if x is not None else None

    def point_on_segment(self, point, seg_start, seg_end):
        """Check if point lies on line segment"""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        
        # Check if point is within bounding box of segment
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        return min_x <= px <= max_x and min_y <= py <= max_y
    
    def clear_point(self, event):
        self.marked_point = None
        self.plot_slice(int(self.slider.val))


    def create_3d_viewer(self):
        """Create and maintain a 3D viewer window showing the meshes and current slice plane"""
        self.plt_3d = Plotter(title="3D View with Slice Plane", size=(800, 600))
        
        # Load meshes
        mesh = trimesh.load(self.model_file)
        prep_mesh = trimesh.load(self.prep_file)
        prep_k_mesh = trimesh.load(self.prep_k_file)
        
        # Apply ICP transformations if available
        tooth_id = f"{v}{xx}"
        icp_filename = f"Zähne/{v}/{xx}/icp_points_{tooth_id}.npy"
        
        if os.path.exists(icp_filename):
            print(f"Loading and applying ICP transformations from {icp_filename}...")
            icp_points = np.load(icp_filename, allow_pickle=True).item()
            source_points = icp_points["source_points"]
            target_points = icp_points["target_points"]
            
            # Compute initial transformation using manually selected points
            initial_T = best_fit_transform(source_points, target_points)
            prep_k_mesh.apply_transform(initial_T)
            
            # Sample points for ICP
            target_points_icp = prep_mesh.sample(5000)
            source_points_icp = prep_k_mesh.sample(5000)
            
            # Run ICP to align präpgrenze_k to präpgrenze
            T, finalA, final_error, iterations = iterative_closest_point(
                source_points_icp, 
                target_points_icp,
                max_iterations=500,
                tolerance=1e-6
            )
            
            # Apply the final ICP transformation
            prep_k_mesh.apply_transform(T)
        
        # Create vedo meshes with the transformed vertices
        self.mesh_vedo = Mesh([mesh.vertices, mesh.faces]).color("green").alpha(0.7)
        self.prep_mesh_vedo = Mesh([prep_mesh.vertices, prep_mesh.faces]).color("red").alpha(0.5)
        self.prep_k_mesh_vedo = Mesh([prep_k_mesh.vertices, prep_k_mesh.faces]).color("blue").alpha(0.5)
        
        # Add meshes to plotter
        self.plt_3d.add(self.mesh_vedo)
        self.plt_3d.add(self.prep_mesh_vedo)
        self.plt_3d.add(self.prep_k_mesh_vedo)
        
        # Create initial slice plane (aligned with X-axis)
        self.slice_plane = self.create_slice_plane(0)
        self.plt_3d.add(self.slice_plane)
        
        # Show in non-blocking mode
        self.plt_3d.show(interactive=False, viewup='z')

    def create_slice_plane(self, angle_idx):
        """Create a plane representing the current slice view"""
        angle = self.angles[angle_idx]
        
        # Calculate center from the mesh vertices
        if hasattr(self, 'mesh_vedo'):
            vertices = self.mesh_vedo.points()
            center = np.mean(vertices, axis=0)
        else:
            center = [0, 0, 0]  # Fallback center
        
        # Create plane perpendicular to X-axis with appropriate size
        bounds_size = self.shared_bounds[1] - self.shared_bounds[0]
        plane_size = np.max(bounds_size) * 2
        
        plane = Plane(
            pos=center,
            normal=(1, 0, 0),
            s=(plane_size, plane_size)
        )
        
        # Rotate around Z axis to match 2D view
        plane.rotate(angle, axis=(0, 0, 1), point=center)
        
        # Style the plane with 20% opacity fill and wireframe
        plane.color('yellow').alpha(0.2).wireframe(False).lighting('ambient')
        return plane

    def update_3d_slice_plane(self, angle_idx):
        """Update the 3D slice plane to match current 2D view"""
        if hasattr(self, 'plt_3d') and self.plt_3d.renderer:
            # Remove old plane if it exists
            if hasattr(self, 'slice_plane'):
                self.plt_3d.remove(self.slice_plane)
                del self.slice_plane
            
            # Create new plane with current angle
            angle = self.angles[angle_idx]
            
            # Calculate center from the mesh vertices
            if hasattr(self, 'mesh_vedo'):
                vertices = self.mesh_vedo.points()
                center = np.mean(vertices, axis=0)
            else:
                center = [0, 0, 0]  # Fallback center
            
            # Create plane perpendicular to X-axis
            bounds_size = self.shared_bounds[1] - self.shared_bounds[0]
            plane_size = np.max(bounds_size) * 2
            
            self.slice_plane = Plane(
                pos=center, 
                normal=(1, 0, 0),
                s=(plane_size, plane_size)
            )
            
            # Rotate around Z axis to match 2D view
            self.slice_plane.rotate(angle, axis=(0, 0, 1), point=center)
            
            # Style the plane with 20% opacity fill (no wireframe)
            self.slice_plane.color('yellow').alpha(0.2).wireframe(False).lighting('ambient')
            
            # Add to plotter and render
            self.plt_3d.add(self.slice_plane)
            self.plt_3d.render()
            
    def save_data(self, event):
        global all_data
        print("Saving data...")
        #csv_filename = f"Zähne/{v}/{xx}/output_{v}_{xx}.csv"
        csv_filename = f"Zähne/Output/{v}_{xx}.csv"
        csv_rows = []

        # Calculate the physical-to-pixel ratio
        with open(csv_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["angle", "x", "y", "distance_mm"])
            for angle in range(0, 365, 5):
                angle_idx = min(angle, self.num_angles - 1)
                #print("angle: ", angle_idx)

                # Get the slice data for the current angle
                slice_data = self.slice_data_cache[angle_idx]
                prep_slice_data = self.prep_slice_data_cache[angle_idx]
                prep_k_slice_data = self.prep_k_slice_data_cache[angle_idx]

                # Apply flip if needed
                if self.is_flipped:
                    slice_data = np.fliplr(np.flipud(slice_data))
                    prep_slice_data = np.fliplr(np.flipud(prep_slice_data))
                    prep_k_slice_data = np.fliplr(np.flipud(prep_k_slice_data))

                # === Use unified calculation method ===
                final_mep, current_sulcusboden, current_vop, extended_line_points, sulcus_line, vop_line, sulcus_distance_mm, vop_distance_mm, sulcustiefe = \
                    self.calculate_mep_and_detection_points(slice_data, prep_slice_data, prep_k_slice_data)


                # Use umschlagpunkt_1, präpgrenze bottom, and präpgrenze_k bottom
                präpgrenze_left = self.find_bottom_left(prep_slice_data)
                präpgrenze_point = self.find_bottom_right(prep_slice_data)
                präpgrenze_k_point = self.find_bottom_right(prep_k_slice_data)

                # Calculate the distance between präpgrenze and präpgrenze_k points in pixels
                MargDev_pixels = None
                MargDev_mm = None
                if präpgrenze_point and präpgrenze_k_point:
                    distance_px = np.sqrt((präpgrenze_k_point[0] - präpgrenze_point[0])**2 + 
                                        (präpgrenze_k_point[1] - präpgrenze_point[1])**2)
                    
                    y_diff = präpgrenze_point[1] - präpgrenze_k_point[1]
                    sign = 1 if y_diff >= 0 else -1
                    # Apply sign while preserving original distance magnitude
                    MargDev_pixels = sign * distance_px
                    MargDev_mm = MargDev_pixels * self.mm_per_pixel

                # Calculate distance from mep to Präpgrenze
                SulcWid_pixels = None
                SulcWid_mm = None
                if final_mep and präpgrenze_k_point:
                    SulcWid_pixels = np.sqrt((final_mep[0] - präpgrenze_k_point[0])**2 +
                                        (final_mep[1] - präpgrenze_k_point[1])**2)
                    SulcWid_mm = SulcWid_pixels * self.mm_per_pixel
                    
                    # Only include if distance <= 5mm
                    if SulcWid_mm < 0.1:
                        final_mep = None
                        SulcWid_pixels = None
                        SulcWid_mm = None
                
                
                # Prepare the row for the CSV
                row = {
                    "angle": angle,
                    "präpgrenze_x": präpgrenze_point[0] if präpgrenze_point else None,
                    "präpgrenze_y": präpgrenze_point[1] if präpgrenze_point else None,
                    "präpgrenze_k_x": präpgrenze_k_point[0] if präpgrenze_k_point else None,
                    "präpgrenze_k_y": präpgrenze_k_point[1] if präpgrenze_k_point else None,
                    "sulcusboden_x": current_sulcusboden[0] if current_sulcusboden else None,
                    "sulcusboden_y": current_sulcusboden[1] if current_sulcusboden else None,
                    "MargExPt_x": final_mep[0] if final_mep else None,
                    "MargExPt_y": final_mep[1] if final_mep else None,
                    "vop_x": current_vop[0] if current_vop else None,
                    "vop_y": current_vop[1] if current_vop else None,
                    #"MargDev_pixels": MargDev_pixels,  # Distance präpgrenze (d) to (k) in pixels
                    "MargDev": MargDev_mm, # Distance in millimeters
                    #"SulcWid": SulcWid_pixels, # Distance mep to Präpgrenze (d)
                    "SulcWid": SulcWid_mm, # in mm
                    "SulcDep": sulcustiefe,
                    "MargExPt_Sulcusboden": sulcus_distance_mm,
                    "vop_MargExPt": vop_distance_mm,
                }
                csv_rows.append(row)
        # Save the data to a CSV file
        with open(csv_filename, mode="w", newline="") as csv_file:
            fieldnames = [
                "angle", "präpgrenze_x", "präpgrenze_y", 
                "präpgrenze_k_x", "präpgrenze_k_y", "sulcusboden_x", "sulcusboden_y",
                "MargExPt_x", "MargExPt_y", "vop_x", "vop_y", "MargDev", "SulcWid",
                "SulcDep", "MargExPt_Sulcusboden", "vop_MargExPt"
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"CSV data saved to {csv_filename}")

        # Save the current correction to all_data
        angle_idx = int(self.slider.val)
        slice_data = self.slice_data_cache[angle_idx]
        prep_slice_data = self.prep_slice_data_cache[angle_idx]
        prep_k_slice_data = self.prep_k_slice_data_cache[angle_idx]

        model_id = os.path.splitext(os.path.basename(self.model_file))[0]

        # Save training_data.npy
        print(f"Data for model '{model_id}' saved")
        
    def run(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.3, top=0.9)
        
        #self.ax.imshow(initial_slice, cmap='bone_r', origin='lower', interpolation='none')
        self.plot_slice(0)
        self.ax.set_title(f'Rotation angle: {self.angles[0]:.2f}° around Z-axis, Slice along X-axis')

        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()

        ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgrey')
        self.slider = Slider(ax_slider, 'Z-rotation angle', 0, self.num_angles - 1, valinit=0, valstep=1)
        self.slider.on_changed(lambda val: self.plot_slice(int(val)))

        ax_save_button = plt.axes([0.8, 0.1, 0.1, 0.03])
        save_button = Button(ax_save_button, 'Save Data')
        save_button.on_clicked(self.save_data)

        ax_flip_button = plt.axes([0.8, 0.15, 0.1, 0.03])
        flip_button = Button(ax_flip_button, 'Flip View 180°')
        flip_button.on_clicked(self.flip_model_180)

        # Connect mouse scroll event for zooming
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        plt.show()
    
    def on_key_press(self, event):
        if event.key == '0':  # Reset zoom
            self.reset_zoom()

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_mouse, y_mouse = event.xdata, event.ydata
        zoom_factor = 1.1 if event.button == 'up' else 0.9
        
        new_xlim = (x_mouse - (x_mouse - xlim[0]) * zoom_factor,
                    x_mouse + (xlim[1] - x_mouse) * zoom_factor)
        new_ylim = (y_mouse - (y_mouse - ylim[0]) * zoom_factor,
                    y_mouse + (ylim[1] - y_mouse) * zoom_factor)
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw()

    def reset_zoom(self):
        """Reset the zoom to the original axis limits."""
        if self.original_xlim is not None and self.original_ylim is not None:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.fig.canvas.draw()

if __name__ == '__main__':
    model_file = f'Zähne/{v}/{xx}/{xx}_umschlagpunkt.stl'
    prep_file = f'Zähne/{v}/{xx}/{xx}_präpgrenze_d.stl'  # Path to the präpgrenze STL file
    prep_k_file = f'Zähne/{v}/{xx}/{xx}_präpgrenze_k.stl' # konv. abformung
    viewer = InteractiveRotationViewer(
        model_file, prep_file, prep_k_file, 
        num_angles=360, 
        slice_resolution=1200, 
        batch_size=60
    )
    viewer.run()
