from datetime import datetime
import math
import random
import numpy as np


PIXELS_PER_INCH = 227 / 2
PX_PER_CM = PIXELS_PER_INCH / 2.54

X_MIN = 0.0
X_MAX = 100.0
Y_MIN = 0.0
Y_MAX = 100.0

SQRT2 = math.sqrt(2.0)
SPACE_CENTER = (50.0, 50.0)
SPACE_DIAG = math.hypot(X_MAX - X_MIN, Y_MAX - Y_MIN)
T_MAJOR = (1.0 / SQRT2, 1.0 / SQRT2)
N_MINOR = (-1.0 / SQRT2, 1.0 / SQRT2)


def now_iso():
    return datetime.now().isoformat()


def to_stim_params(x, y):
    xt_cycles_per_cm = (x * 5.0) / 100.0
    sf = xt_cycles_per_cm / PX_PER_CM
    ori_deg = (y * 90.0) / 100.0
    return sf, ori_deg


def key_to_interval(raw_key):
    if raw_key in {"1", "num_1"}:
        return 1
    if raw_key in {"2", "num_2"}:
        return 2
    return None


def signed_boundary_distance(x, y):
    return (y - x) / SQRT2


def make_cp_geometry(axis_gap=30.0, major_axis_frac=0.75, minor_axis_len=25.0):
    major_len = max(0.05, min(0.95, major_axis_frac)) * SPACE_DIAG
    half_major = major_len * 0.5
    half_minor = max(1.0, min(minor_axis_len * 0.5, SPACE_DIAG * 0.45))
    half_gap = axis_gap * 0.5
    center_a = {
        "x": SPACE_CENTER[0] + N_MINOR[0] * half_gap,
        "y": SPACE_CENTER[1] + N_MINOR[1] * half_gap,
    }
    center_b = {
        "x": SPACE_CENTER[0] - N_MINOR[0] * half_gap,
        "y": SPACE_CENTER[1] - N_MINOR[1] * half_gap,
    }
    return {
        "half_major": half_major,
        "half_minor": half_minor,
        "center_a": center_a,
        "center_b": center_b,
    }


def build_cp_trial_runtime(cell, geometry, rng):
    def shift_point(pt, direction, amount):
        return {
            "x": pt["x"] + direction[0] * amount,
            "y": pt["y"] + direction[1] * amount,
        }

    def point_on_side(pt, category):
        d = signed_boundary_distance(pt["x"], pt["y"])
        return d > 0 if category == "A" else d < 0

    def point_in_category_ellipse(pt, category):
        if category == "A":
            center = geometry["center_a"]
        else:
            center = geometry["center_b"]
        rel_x = pt["x"] - center["x"]
        rel_y = pt["y"] - center["y"]
        major_coord = rel_x * T_MAJOR[0] + rel_y * T_MAJOR[1]
        minor_coord = rel_x * N_MINOR[0] + rel_y * N_MINOR[1]
        q = (
            (major_coord * major_coord) / (geometry["half_major"] * geometry["half_major"])
            + (minor_coord * minor_coord) / (geometry["half_minor"] * geometry["half_minor"])
        )
        return q <= 1.0 + 1e-9

    def point_in_category(pt, category):
        return (
            X_MIN <= pt["x"] <= X_MAX
            and Y_MIN <= pt["y"] <= Y_MAX
            and point_on_side(pt, category)
            and point_in_category_ellipse(pt, category)
        )

    def sample_point_in_category(category):
        if category == "A":
            center = geometry["center_a"]
        else:
            center = geometry["center_b"]
        for _ in range(500):
            r = math.sqrt(rng.random())
            theta = rng.uniform(0.0, 2.0 * math.pi)
            major_coord = geometry["half_major"] * r * math.cos(theta)
            minor_coord = geometry["half_minor"] * r * math.sin(theta)
            pt = {
                "x": center["x"] + T_MAJOR[0] * major_coord + N_MINOR[0] * minor_coord,
                "y": center["y"] + T_MAJOR[1] * major_coord + N_MINOR[1] * minor_coord,
            }
            if point_in_category(pt, category):
                return pt
        return {"x": center["x"], "y": center["y"]}

    if cell["family"] in {"within_A", "within_B"}:
        category = "A" if cell["family"] == "within_A" else "B"
        pair = None
        for _ in range(500):
            center = sample_point_in_category(category)
            p1 = shift_point(center, T_MAJOR, cell["distance"] * 0.5)
            p2 = shift_point(center, T_MAJOR, -cell["distance"] * 0.5)
            if point_in_category(p1, category) and point_in_category(p2, category):
                pair = {"ref": p1, "cmp": p2}
                break
        if pair is None:
            center = sample_point_in_category(category)
            pair = {"ref": center, "cmp": shift_point(center, T_MAJOR, 0.01)}
    else:
        pair = None
        for _ in range(500):
            t = rng.uniform(0.15, 0.85)
            mid = {
                "x": X_MIN + (X_MAX - X_MIN) * t,
                "y": Y_MIN + (Y_MAX - Y_MIN) * t,
            }
            p_a = shift_point(mid, N_MINOR, cell["distance"] * 0.5)
            p_b = shift_point(mid, N_MINOR, -cell["distance"] * 0.5)
            if point_in_category(p_a, "A") and point_in_category(p_b, "B"):
                pair = {"ref": p_a, "cmp": p_b}
                break
        if pair is None:
            pair = {
                "ref": shift_point({"x": SPACE_CENTER[0], "y": SPACE_CENTER[1]}, N_MINOR, 0.5),
                "cmp": shift_point({"x": SPACE_CENTER[0], "y": SPACE_CENTER[1]}, N_MINOR, -0.5),
            }

    diff_interval = 1 if rng.random() < 0.5 else 2
    flip_order = rng.random() < 0.5
    same_pair = {"a": pair["ref"], "b": pair["ref"]}
    if flip_order:
        diff_pair = {"a": pair["cmp"], "b": pair["ref"]}
    else:
        diff_pair = {"a": pair["ref"], "b": pair["cmp"]}
    int1 = diff_pair if diff_interval == 1 else same_pair
    int2 = diff_pair if diff_interval == 2 else same_pair

    if cell["family"] == "within_A":
        pair_type = "within"
    elif cell["family"] == "within_B":
        pair_type = "within"
    else:
        pair_type = "across"

    return {
        "condition_id": cell["condition_id"],
        "cp_family": cell["family"],
        "cp_distance_level": cell["distance_level"],
        "distance": cell["distance"],
        "pair_type": pair_type,
        "diff_interval": diff_interval,
        "int1a": int1["a"],
        "int1b": int1["b"],
        "int2a": int2["a"],
        "int2b": int2["b"],
    }


def make_rsa_pool_centered_grid(grid_n=7,
                                x_min=20.0,
                                x_max=100.0,
                                y_min=0.0,
                                y_max=100.0):
    if grid_n <= 1:
        raise ValueError("grid_n must be > 1")
    xs = np.linspace(x_min, x_max, grid_n)
    ys = np.linspace(y_min, y_max, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    item_ids = np.arange(x_flat.size, dtype=int)
    pool = []
    for item_id, x, y in zip(item_ids.tolist(), x_flat.tolist(), y_flat.tolist()):
        pool.append({"item_id": item_id, "x": float(x), "y": float(y)})
    return pool


def _assign_extra_blocks_exact(n_items, n_blocks, extras_per_item, extras_targets, rng):
    if len(extras_targets) != n_blocks:
        raise ValueError("extras_targets length mismatch")

    for _ in range(200):
        remaining = list(extras_targets)
        assignment = [[] for _ in range(n_items)]
        item_order = list(range(n_items))
        rng.shuffle(item_order)
        ok = True

        for item in item_order:
            chosen = []
            for _k in range(extras_per_item):
                candidates = [b for b in range(n_blocks) if remaining[b] > 0 and b not in chosen]
                if not candidates:
                    ok = False
                    break
                max_remaining = max(remaining[b] for b in candidates)
                top = [b for b in candidates if remaining[b] == max_remaining]
                b = rng.choice(top)
                chosen.append(b)
                remaining[b] -= 1
            if not ok:
                break
            assignment[item] = chosen

        if ok and all(v == 0 for v in remaining):
            return assignment

    raise RuntimeError("Failed to build exact RSA extras-by-block assignment.")


def _reduce_adjacent_item_repeats(item_ids):
    for i in range(1, len(item_ids)):
        if item_ids[i] != item_ids[i - 1]:
            continue
        swap_j = None
        for j in range(i + 1, len(item_ids)):
            if item_ids[j] != item_ids[i - 1] and (
                j == len(item_ids) - 1 or item_ids[j] != item_ids[j + 1]
            ):
                swap_j = j
                break
        if swap_j is None:
            for j in range(i + 1, len(item_ids)):
                if item_ids[j] != item_ids[i - 1]:
                    swap_j = j
                    break
        if swap_j is not None:
            item_ids[i], item_ids[swap_j] = item_ids[swap_j], item_ids[i]


def make_rsa_schedule(pool, repeats_per_item=20, n_blocks=8, schedule_seed="rsa_schedule"):
    pool_size = len(pool)
    total_trials = pool_size * repeats_per_item
    rng = random.Random(schedule_seed)

    base_per_block = repeats_per_item // n_blocks
    extras_per_item = repeats_per_item % n_blocks
    total_extras = pool_size * extras_per_item
    extras_per_block_base = total_extras // n_blocks
    extras_per_block_rem = total_extras % n_blocks
    extras_targets = [
        extras_per_block_base + (1 if b < extras_per_block_rem else 0)
        for b in range(n_blocks)
    ]
    rng.shuffle(extras_targets)

    extras_assignment = _assign_extra_blocks_exact(
        n_items=pool_size,
        n_blocks=n_blocks,
        extras_per_item=extras_per_item,
        extras_targets=extras_targets,
        rng=rng,
    )

    blocks = [[] for _ in range(n_blocks)]
    for item_id in range(pool_size):
        extra_set = set(extras_assignment[item_id])
        for block_index in range(n_blocks):
            n_here = base_per_block + (1 if block_index in extra_set else 0)
            blocks[block_index].extend([item_id] * n_here)

    expected_sizes = [
        total_trials // n_blocks + (1 if b < (total_trials % n_blocks) else 0)
        for b in range(n_blocks)
    ]
    expected_sizes.sort(reverse=True)
    actual_sizes = sorted([len(block) for block in blocks], reverse=True)
    if actual_sizes != expected_sizes:
        raise ValueError(
            f"RSA block size mismatch: expected multiset {expected_sizes}, got {actual_sizes}"
        )

    for block_index in range(n_blocks):
        rng.shuffle(blocks[block_index])
        _reduce_adjacent_item_repeats(blocks[block_index])

    return blocks
