import math
import random
from datetime import datetime
import numpy as np
import pandas as pd
from psychopy import core, event, visual

SQRT2 = math.sqrt(2.0)
SPACE_CENTER = (50.0, 50.0)
X_MIN = 0.0
X_MAX = 100.0
Y_MIN = 0.0
Y_MAX = 100.0
SPACE_DIAG = math.hypot(X_MAX - X_MIN, Y_MAX - Y_MIN)
T_MAJOR = (1.0 / SQRT2, 1.0 / SQRT2)
N_MINOR = (-1.0 / SQRT2, 1.0 / SQRT2)


def transform_stim(x, y):
    # xt maps x from [0, 100] to [0, 5]
    # yt maps y from [0, 100] to [0, 90] degrees
    xt = np.asarray(x, dtype=float) * 5.0 / 100.0
    yt = np.asarray(y, dtype=float) * 90.0 / 100.0
    return xt, yt


def stim_xy_to_sf_ori_deg(x, y, px_per_cm):
    xt, yt = transform_stim(x, y)
    sf = np.asarray(xt, dtype=float) / px_per_cm
    ori_deg = np.asarray(yt, dtype=float)
    return sf, ori_deg


def make_stim_cats(n_stimuli_per_category=2000, random_seed=None):
    rng = np.random.default_rng(random_seed)

    # Define covariance matrix parameters
    var = 100
    corr = 0.9
    sigma = np.sqrt(var)

    # Rotation matrix
    theta = 45 * np.pi / 180
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # Means for the two categories
    category_A_mean = [40, 60]
    category_B_mean = [60, 40]

    # Standard deviations along major and minor axes
    std_major = sigma * np.sqrt(1 + corr)
    std_minor = sigma * np.sqrt(1 - corr)

    def sample_within_ellipse(mean, n_samples):

        # Sample radius
        r = np.sqrt(rng.uniform(
            0, 9, n_samples))  # 3 standard deviations, squared is 9

        # Sample angle
        angle = rng.uniform(0, 2 * np.pi, n_samples)

        # Convert polar to Cartesian coordinates
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Scale by standard deviations
        x_scaled = x * std_major
        y_scaled = y * std_minor

        # Apply rotation
        points = np.dot(rotation_matrix, np.vstack([x_scaled, y_scaled]))

        # Translate to mean
        points[0, :] += mean[0]
        points[1, :] += mean[1]

        return points.T

    # Generate stimuli
    stimuli_A = sample_within_ellipse(category_A_mean, n_stimuli_per_category)
    stimuli_B = sample_within_ellipse(category_B_mean, n_stimuli_per_category)

    # Define labels to match runtime response labels.
    labels_A = np.array(["A"] * n_stimuli_per_category)
    labels_B = np.array(["B"] * n_stimuli_per_category)

    # Concatenate the stimuli and labels
    stimuli = np.concatenate([stimuli_A, stimuli_B])
    labels = np.concatenate([labels_A, labels_B])

    # Put the stimuli and labels together into a dataframe
    ds = pd.DataFrame({"x": stimuli[:, 0], "y": stimuli[:, 1], "cat": labels})

    # shuffle rows of ds
    if random_seed is None:
        ds = ds.sample(frac=1).reset_index(drop=True)
    else:
        ds = ds.sample(frac=1, random_state=int(rng.integers(
            0, 2**32 - 1))).reset_index(drop=True)

    # create 90 degree rotation stim
    ds_90 = ds.copy()
    ds_90["x"] = ds_90["x"] - 50
    ds_90["y"] = ds_90["y"] - 50
    theta = 90 * np.pi / 180
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    rotated_points = np.dot(rotation_matrix, ds_90[["x", "y"]].T).T
    ds_90["x"] = rotated_points[:, 0]
    ds_90["y"] = rotated_points[:, 1]
    ds_90["x"] = ds_90["x"] + 50
    ds_90["y"] = ds_90["y"] + 50

    # create 180 degree rotation stim
    ds_180 = ds.copy()
    ds_180["x"] = ds_180["x"] - 50
    ds_180["y"] = ds_180["y"] - 50
    theta = 180 * np.pi / 180
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    rotated_points = np.dot(rotation_matrix, ds_180[["x", "y"]].T).T
    ds_180["x"] = rotated_points[:, 0]
    ds_180["y"] = rotated_points[:, 1]
    ds_180["x"] = ds_180["x"] + 50
    ds_180["y"] = ds_180["y"] + 50

    return ds, ds_90, ds_180


def create_grating_patch(size, freq, theta):
    """
    Generate a grating patch with a circular mask using NumPy.
    The units of size are pixels, the units of freq are
    cycles per pixel, and the units of theta are radians.
    """
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    x, y = np.meshgrid(x, y)

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # grating formula
    psi = 0
    gb = np.cos(2 * np.pi * freq * x_theta + psi)

    # Circular mask
    radius = size / 2
    circle_mask = (x**2 + y**2) <= radius**2
    gb *= circle_mask

    return gb


def plot_stim_space_examples(ds,
                             win,
                             grating,
                             px_per_cm,
                             x_col="x",
                             y_col="y",
                             n_examples=6):

    ds_plot = ds[[x_col,
                  y_col]].dropna().drop_duplicates().reset_index(drop=True)
    ds_plot = ds_plot.sort_values([x_col, y_col]).reset_index(drop=True)

    sample_idx = np.linspace(0, len(ds) - 1, n_examples, dtype=int)
    sample_idx = np.unique(sample_idx)
    ds_plot = ds.iloc[sample_idx].reset_index(drop=True)

    screen_h = win.size[1]
    x_span = 100.0
    y_span = 100.0
    inner_scale = 0.5

    stim_objs = []
    for _, row in ds_plot.iterrows():
        sf, ori_deg = stim_xy_to_sf_ori_deg(row[x_col], row[y_col], px_per_cm)
        stim = visual.GratingStim(
            win,
            tex=grating.tex,
            mask=grating.mask,
            texRes=grating.texRes,
            interpolate=grating.interpolate,
            size=grating.size,
            units=grating.units,
            sf=float(np.asarray(sf)),
            ori=float(np.asarray(ori_deg)),
        )
        x_pix = ((row[x_col]) / x_span - 0.5) * screen_h * inner_scale
        y_pix = ((row[y_col]) / y_span - 0.5) * screen_h * inner_scale
        stim.pos = (x_pix, y_pix)
        stim_objs.append(stim)

    event.clearEvents()

    while True:
        for stim in stim_objs:
            stim.draw()
        win.flip()

        keys = event.getKeys()
        if "escape" in keys:
            win.close()
            core.quit()
        if "space" in keys:
            break


def make_rsa_pool_grid(grid_n=7, x_min=4.0, x_max=96.0, y_min=4.0, y_max=96.0):

    xs = np.linspace(x_min, x_max, grid_n)
    ys = np.linspace(y_min, y_max, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    ds = pd.DataFrame({
        "item_id": np.arange(xx.size, dtype=int),
        "x": xx.ravel(),
        "y": yy.ravel(),
    })
    return ds


def make_rsa_schedule_table(pool,
                            repeats_per_block=2,
                            n_blocks=8,
                            schedule_seed="rsa_schedule"):

    rng = random.Random(str(schedule_seed))
    block_tables = []

    for block_id in range(1, n_blocks + 1):
        copies = []
        last_item_id = None

        for _ in range(repeats_per_block):
            while True:
                seed = rng.randrange(0, 2**32)
                shuffled = pool.sample(
                    frac=1, random_state=seed).reset_index(drop=True)
                if last_item_id is None or int(
                        shuffled.iloc[0]["item_id"]) != last_item_id:
                    break
            copies.append(shuffled)
            last_item_id = int(shuffled.iloc[-1]["item_id"])

        block_df = pd.concat(copies, ignore_index=True)
        block_df["block_id"] = block_id
        block_df["block_trial"] = np.arange(1, len(block_df) + 1, dtype=int)
        block_tables.append(block_df)

    schedule = pd.concat(block_tables, ignore_index=True)
    schedule["trial"] = np.arange(len(schedule), dtype=int)

    return schedule


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


def make_cp_trial_table(practice_far_n=16,
                        practice_moderate_n=8,
                        main_reps_per_cell=34,
                        near_dist=6.0,
                        far_dist=15.0,
                        schedule_seed="cp_schedule"):
    rng = random.Random(schedule_seed)
    families = ["within_A", "within_B", "between_AB"]
    moderate_dist = 0.5 * (near_dist + far_dist)

    practice_rows = []
    for level, dist, n_trials in [
        ("far", float(far_dist), practice_far_n),
        ("moderate", float(moderate_dist), practice_moderate_n),
    ]:
        base = n_trials // len(families)
        rem = n_trials % len(families)
        fam_order = families[:]
        rng.shuffle(fam_order)
        for i, fam in enumerate(fam_order):
            n_here = base + (1 if i < rem else 0)
            for _ in range(n_here):
                practice_rows.append({
                    "phase": "practice",
                    "family": fam,
                    "distance_level": level,
                    "distance": dist,
                    "condition_id": f"practice_{fam}_{level}_{dist:.3f}",
                    "block_id": 0,
                })
    rng.shuffle(practice_rows)

    main_rows = []
    for family in families:
        for distance_level, distance in [("near", near_dist),
                                         ("far", far_dist)]:
            for _ in range(main_reps_per_cell):
                main_rows.append({
                    "phase": "main",
                    "family": family,
                    "distance_level": distance_level,
                    "distance": float(distance),
                    "condition_id":
                    f"{family}_{distance_level}_{distance:.3f}",
                    "block_id": 1,
                })
    rng.shuffle(main_rows)

    ds = pd.DataFrame(practice_rows + main_rows)
    ds["trial"] = np.arange(len(ds), dtype=int)
    return ds


def make_cp_pair_tables(n_stimuli_per_category=400, pool_seed="cp_pool"):

    pool_seed_int = random.Random(str(pool_seed)).randrange(0, 2**32)
    ds, _, _ = make_stim_cats(
        n_stimuli_per_category=n_stimuli_per_category,
        random_seed=pool_seed_int,
    )
    ds["stim_id"] = np.arange(len(ds), dtype=int)

    def build_pair_table(ds_left, ds_right, allow_same_category):
        xy_left = ds_left[["x", "y"]].to_numpy(dtype=float)
        xy_right = ds_right[["x", "y"]].to_numpy(dtype=float)
        stim_left = ds_left["stim_id"].to_numpy(dtype=int)
        stim_right = ds_right["stim_id"].to_numpy(dtype=int)

        if allow_same_category:
            idx_left, idx_right = np.triu_indices(len(ds_left), k=1)
            pts_left = xy_left[idx_left]
            pts_right = xy_right[idx_right]
            stim_id_left = stim_left[idx_left]
            stim_id_right = stim_right[idx_right]
        else:
            idx_left, idx_right = np.indices((len(ds_left), len(ds_right)))
            idx_left = idx_left.ravel()
            idx_right = idx_right.ravel()
            pts_left = xy_left[idx_left]
            pts_right = xy_right[idx_right]
            stim_id_left = stim_left[idx_left]
            stim_id_right = stim_right[idx_right]

        dist = np.linalg.norm(pts_left - pts_right, axis=1)
        return pd.DataFrame({
            "ref_id": stim_id_left,
            "cmp_id": stim_id_right,
            "ref_x": pts_left[:, 0],
            "ref_y": pts_left[:, 1],
            "cmp_x": pts_right[:, 0],
            "cmp_y": pts_right[:, 1],
            "distance": dist,
        })

    ds_a = ds.loc[ds["cat"] == "A"].reset_index(drop=True)
    ds_b = ds.loc[ds["cat"] == "B"].reset_index(drop=True)

    return {
        "within_A": build_pair_table(ds_a, ds_a, allow_same_category=True),
        "within_B": build_pair_table(ds_b, ds_b, allow_same_category=True),
        "between_AB": build_pair_table(ds_a, ds_b, allow_same_category=False),
    }


def signed_boundary_distance(x, y):
    return (y - x) / SQRT2


def key_to_interval(raw_key):
    if raw_key in {"1", "num_1"}:
        return 1
    if raw_key in {"2", "num_2"}:
        return 2
    return None


def build_cp_trial_runtime(trial_row, geometry, rng):

    def shift_point(pt, direction, amount):
        return {
            "x": pt["x"] + direction[0] * amount,
            "y": pt["y"] + direction[1] * amount,
        }

    def point_on_side(pt, category):
        d = signed_boundary_distance(pt["x"], pt["y"])
        return d > 0 if category == "A" else d < 0

    def point_in_category_ellipse(pt, category):
        center = geometry["center_a"] if category == "A" else geometry[
            "center_b"]
        rel_x = pt["x"] - center["x"]
        rel_y = pt["y"] - center["y"]
        major_coord = rel_x * T_MAJOR[0] + rel_y * T_MAJOR[1]
        minor_coord = rel_x * N_MINOR[0] + rel_y * N_MINOR[1]
        q = ((major_coord * major_coord) /
             (geometry["half_major"] * geometry["half_major"]) +
             (minor_coord * minor_coord) /
             (geometry["half_minor"] * geometry["half_minor"]))
        return q <= 1.0 + 1e-9

    def point_in_category(pt, category):
        return (X_MIN <= pt["x"] <= X_MAX and Y_MIN <= pt["y"] <= Y_MAX
                and point_on_side(pt, category)
                and point_in_category_ellipse(pt, category))

    def sample_point_in_category(category):
        center = geometry["center_a"] if category == "A" else geometry[
            "center_b"]
        for _ in range(500):
            r = math.sqrt(rng.random())
            theta = rng.uniform(0.0, 2.0 * math.pi)
            major_coord = geometry["half_major"] * r * math.cos(theta)
            minor_coord = geometry["half_minor"] * r * math.sin(theta)
            pt = {
                "x":
                center["x"] + T_MAJOR[0] * major_coord +
                N_MINOR[0] * minor_coord,
                "y":
                center["y"] + T_MAJOR[1] * major_coord +
                N_MINOR[1] * minor_coord,
            }
            if point_in_category(pt, category):
                return pt
        return {"x": center["x"], "y": center["y"]}

    family = trial_row["family"]
    distance = float(trial_row["distance"])

    if family in {"within_A", "within_B"}:
        category = "A" if family == "within_A" else "B"
        pair = None
        for _ in range(500):
            center = sample_point_in_category(category)
            p1 = shift_point(center, T_MAJOR, distance * 0.5)
            p2 = shift_point(center, T_MAJOR, -distance * 0.5)
            if point_in_category(p1, category) and point_in_category(
                    p2, category):
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
            p_a = shift_point(mid, N_MINOR, distance * 0.5)
            p_b = shift_point(mid, N_MINOR, -distance * 0.5)
            if point_in_category(p_a, "A") and point_in_category(p_b, "B"):
                pair = {"ref": p_a, "cmp": p_b}
                break
        if pair is None:
            pair = {
                "ref":
                shift_point({
                    "x": SPACE_CENTER[0],
                    "y": SPACE_CENTER[1]
                }, N_MINOR, 0.5),
                "cmp":
                shift_point({
                    "x": SPACE_CENTER[0],
                    "y": SPACE_CENTER[1]
                }, N_MINOR, -0.5),
            }

    diff_interval = 1 if rng.random() < 0.5 else 2
    flip_order = rng.random() < 0.5
    same_pair = {"a": pair["ref"], "b": pair["ref"]}
    diff_pair = {
        "a": pair["cmp"],
        "b": pair["ref"]
    } if flip_order else {
        "a": pair["ref"],
        "b": pair["cmp"]
    }
    int1 = diff_pair if diff_interval == 1 else same_pair
    int2 = diff_pair if diff_interval == 2 else same_pair

    pair_type = "within" if family in {"within_A", "within_B"} else "across"

    return {
        "condition_id": trial_row["condition_id"],
        "cp_family": family,
        "cp_distance_level": trial_row["distance_level"],
        "distance": distance,
        "pair_type": pair_type,
        "diff_interval": diff_interval,
        "int1a": int1["a"],
        "int1b": int1["b"],
        "int2a": int2["a"],
        "int2b": int2["b"],
    }


def build_cp_trial_runtime_from_pairs(trial_row,
                                      pair_tables,
                                      rng,
                                      fallback_n=256):
    family = trial_row["family"]
    pair_table = pair_tables[family]
    distance_values = pair_table["distance"].to_numpy(dtype=float)
    distance_level = trial_row["distance_level"]

    if distance_level == "near":
        lower_q, upper_q = 0.00, 0.20
    elif distance_level == "moderate":
        lower_q, upper_q = 0.40, 0.60
    elif distance_level == "far":
        lower_q, upper_q = 0.80, 1.00
    else:
        lower_q, upper_q = 0.00, 1.00

    lower = float(np.quantile(distance_values, lower_q))
    upper = float(np.quantile(distance_values, upper_q))
    candidate_idx = np.flatnonzero((distance_values >= lower)
                                   & (distance_values <= upper))

    if candidate_idx.size == 0:
        band_center = 0.5 * (lower + upper)
        dist_delta = np.abs(distance_values - band_center)
        candidate_idx = np.argsort(
            dist_delta)[:min(fallback_n, len(pair_table))]

    pair_row = pair_table.iloc[int(candidate_idx[rng.randrange(
        len(candidate_idx))])]
    ref = {"x": float(pair_row["ref_x"]), "y": float(pair_row["ref_y"])}
    cmp = {"x": float(pair_row["cmp_x"]), "y": float(pair_row["cmp_y"])}

    if rng.random() < 0.5:
        ref, cmp = cmp, ref

    diff_interval = 1 if rng.random() < 0.5 else 2
    flip_order = rng.random() < 0.5
    same_ref = ref if rng.random() < 0.5 else cmp
    same_pair = {"a": same_ref, "b": same_ref}
    diff_pair = {"a": cmp, "b": ref} if flip_order else {"a": ref, "b": cmp}
    int1 = diff_pair if diff_interval == 1 else same_pair
    int2 = diff_pair if diff_interval == 2 else same_pair

    pair_type = "within" if family in {"within_A", "within_B"} else "across"

    return {
        "condition_id": trial_row["condition_id"],
        "cp_family": family,
        "cp_distance_level": distance_level,
        "distance": float(pair_row["distance"]),
        "distance_target": float(trial_row["distance"]),
        "pair_type": pair_type,
        "diff_interval": diff_interval,
        "int1a": int1["a"],
        "int1b": int1["b"],
        "int2a": int2["a"],
        "int2b": int2["b"],
    }
