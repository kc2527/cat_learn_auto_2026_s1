import numpy as np
import pandas as pd
from psychopy import core, event, visual


def transorm_stim(x, y):
    # xt maps x from [0, 100] to [0, 5]
    # yt maps y from [0, 100] to [0, 90]
    xt = x * 5 / 100
    yt = (y * 90 / 100) * np.pi / 180
    return xt, yt


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
    xt, yt = transorm_stim(ds["x"], ds["y"])
    ds["xt"] = xt
    ds["yt"] = yt

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


def plot_stim_space_examples(ds, win, grating, px_per_cm):

    screen_h = win.size[1]

    x_span = 100
    y_span = 100
    inner_scale = 0.5

    rows = []
    n_per_cat = 3

    x = np.array([25, 50, 75])
    x_A = x - 10
    x_B = x + 10
    y_A = x_A + 20
    y_B = x_B - 20

    x = np.concatenate([x_A, x_B])
    y = np.concatenate([y_A, y_B])
    xt, yt = transorm_stim(x, y)
    cat = np.array(["A"] * 3 + ["B"] * 3)

    stim_objs = []
    for i in range(len(x)):
        stim = visual.GratingStim(
            win,
            tex=grating.tex,
            mask=grating.mask,
            texRes=grating.texRes,
            interpolate=grating.interpolate,
            size=grating.size,
            units=grating.units,
            sf= xt[i] / px_per_cm,
            ori= yt[i] * 180.0 / np.pi,
        )
        # screen center is (0, 0)
        x_pix = ((x[i]) / x_span - 0.5) * screen_h * inner_scale
        y_pix = ((y[i]) / y_span - 0.5) * screen_h * inner_scale
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
