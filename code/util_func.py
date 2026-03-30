from imports import *


def make_stim_cats(n_stimuli_per_category=2000):

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
        r = np.sqrt(np.random.uniform(
            0, 9, n_samples))  # 3 standard deviations, squared is 9

        # Sample angle
        angle = np.random.uniform(0, 2 * np.pi, n_samples)

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

    # Add a transformed version of the stimuli
    # let xt map x from [0, 100] to [0, 5]
    # let yt map y from [0, 100] to [0, 90]
    ds["xt"] = ds["x"] * 5 / 100
    ds["yt"] = (ds["y"] * 90 / 100) * np.pi / 180

    # shuffle rows of ds
    ds = ds.sample(frac=1).reset_index(drop=True)

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

#    fig, ax = plt.subplots(1, 3, squeeze=False,  figsize=(12, 6))
#    sns.scatterplot(data=ds, x="x", y="y", hue="cat", alpha=0.5, ax=ax[0, 0])
#    sns.scatterplot(data=ds_90, x="x", y="y", hue="cat", alpha=0.5, ax=ax[0, 1])
#    sns.scatterplot(data=ds_180, x="x", y="y", hue="cat", alpha=0.5, ax=ax[0, 2])
#    plt.tight_layout()
#    plt.show()

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


def grating_to_surface(grating_patch):
    import pygame

    normalized_patch = (grating_patch + 1) / 2 * 255
    uint8_patch = normalized_patch.astype(np.uint8)
    surface = pygame.Surface((grating_patch.shape[0], grating_patch.shape[1]),
                             pygame.SRCALPHA)
    pygame.surfarray.blit_array(surface, np.dstack([uint8_patch] * 3))
    return surface


def plot_stim_space_examples(ds=None, win=None):
    from psychopy import visual, event, core  # type: ignore

    if ds is None:
        ds, _, _ = make_stim_cats(n_stimuli_per_category=200)

    owns_window = win is None
    if owns_window:
        win = visual.Window(size=(1400, 900),
                            fullscr=False,
                            units='pix',
                            color=(0.494, 0.494, 0.494),
                            colorSpace='rgb',
                            useRetina=False,
                            waitBlanking=True)

    pixels_per_inch = 227 / 2
    px_per_cm = pixels_per_inch / 2.54
    size_cm = 3
    size_px = int(size_cm * px_per_cm)

    # Show representative real stimuli from each category in their abstract sf/orientation space.
    exemplars = []
    for cat in ["A", "B"]:
        dd = ds[ds["cat"] == cat].copy().sort_values(["xt", "yt"]).reset_index(drop=True)
        if dd.empty:
            continue
        idx = np.linspace(0, len(dd) - 1, 3).round().astype(int)
        exemplars.extend(dd.iloc[idx].to_dict("records"))

    xt_min, xt_max = ds["xt"].min(), ds["xt"].max()
    yt_min, yt_max = ds["yt"].min(), ds["yt"].max()

    def map_range(val, lo, hi, out_lo, out_hi):
        if np.isclose(hi, lo):
            return (out_lo + out_hi) / 2
        return out_lo + ((val - lo) / (hi - lo)) * (out_hi - out_lo)

    old_color = win.color
    win.color = (0.494, 0.494, 0.494)

    header = visual.TextStim(
        win,
        text=("Stimulus check\n"
              "Exemplars are placed by their stimulus-space coordinates:\n"
              "Press SPACE to continue or ESC to quit."),
        color='white',
        height=28,
        pos=(0, 360),
        wrapWidth=1500,
    )

    event.clearEvents()

    stim_objs = []
    label_objs = []

    for ex in exemplars:
        pos = (
            map_range(ex["xt"], xt_min, xt_max, -240, 240),
            map_range(ex["yt"], yt_min, yt_max, -240, 240),
        )

        stim_objs.append(
            visual.GratingStim(
                win,
                tex='sin',
                mask='circle',
                size=(size_px, size_px),
                units='pix',
                pos=pos,
                sf=ex["xt"] * px_per_cm**-1,
                ori=np.degrees(ex["yt"]),
                interpolate=True,
            ))
        label_objs.append(
            visual.TextStim(
                win,
                text=(
                    f"Cat {ex['cat']}\n"
                    f"x={ex['x']:.1f}, y={ex['y']:.1f}\n"
                    f"sf={ex['xt']:.2f}, ori={np.degrees(ex['yt']):.1f}°"
                ),
                color='lightgreen' if ex["cat"] == "A" else 'salmon',
                height=20,
                pos=(pos[0], pos[1] - size_px / 2 - 55),
            ))

    while True:
        header.draw()
        for stim in stim_objs:
            stim.draw()
        for label in label_objs:
            label.draw()
        win.flip()

        keys = event.getKeys()
        if "escape" in keys:
            if owns_window:
                win.close()
            core.quit()
        if "space" in keys:
            break

    win.color = old_color
    if owns_window:
        win.close()
