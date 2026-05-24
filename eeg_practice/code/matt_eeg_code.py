"""
EEG analysis tutorial: a simple pipeline from raw EEG to event-related
potentials for one representative participant.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import mne
import numpy as np
import pandas as pd

os.makedirs("figures", exist_ok=True)


# ---------------------------------------------------------------------
# 1. Event codes and plot colours
# ---------------------------------------------------------------------

stimulus_codes = [20, 21, 22, 23]
response_codes = [30, 31, 32, 33]
feedback_codes = [40, 41, 42, 43]

stimulus_color = "#1f77b4"
response_color = "#2ca02c"
feedback_color = "#d62728"


# ---------------------------------------------------------------------
# 2. BioSemi channel names
# ---------------------------------------------------------------------

# The BDF file uses BioSemi's own labels (A1–A32, B1–B32).
# We build the matching list of standard MNE names so we can rename
# them and then apply a montage with known electrode positions.

raw_scalp_channels = []
for i in range(1, 33):
    raw_scalp_channels.append(f"A{i}")
for i in range(1, 33):
    raw_scalp_channels.append(f"B{i}")

biosemi64_channels = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7",
    "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7",
    "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
    "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz",
    "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4",
    "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz",
    "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]


# ---------------------------------------------------------------------
# 3. Load the raw EEG and behavioural table
# ---------------------------------------------------------------------

raw = mne.io.read_raw_bdf("teaching_eeg_001.bdf", preload=True, verbose="error")
behaviour = pd.read_csv("teaching_behaviour_001.csv")

original_sampling_rate = raw.info["sfreq"]
recording_duration = raw.n_times / original_sampling_rate


# ---------------------------------------------------------------------
# 4. Find event markers
# ---------------------------------------------------------------------

events = mne.find_events(raw, stim_channel=None, shortest_event=1, verbose="error")
event_codes, event_counts = np.unique(events[:, 2], return_counts=True)


# ---------------------------------------------------------------------
# 5. Label channel types and rename to standard MNE names
# ---------------------------------------------------------------------

# Mark auxiliary channels (EXG, GSR, Erg, Resp, Plet, Temp) as misc
# so they are excluded from the average reference computation.
aux_channel_types = {}
for channel_name in raw.ch_names:
    if channel_name != "Status" and channel_name not in raw_scalp_channels:
        aux_channel_types[channel_name] = "misc"
raw.set_channel_types(aux_channel_types, on_unit_change="ignore")

scalp_channel_types = {}
for channel_name in raw_scalp_channels:
    scalp_channel_types[channel_name] = "eeg"
raw.set_channel_types(scalp_channel_types)

rename_map = {}
for i in range(len(raw_scalp_channels)):
    rename_map[raw_scalp_channels[i]] = biosemi64_channels[i]
raw.rename_channels(rename_map)

raw.set_montage(mne.channels.make_standard_montage("biosemi64"), on_missing="ignore")


# ---------------------------------------------------------------------
# 6. Preprocessing
# ---------------------------------------------------------------------

raw.set_eeg_reference("average", projection=False, verbose="error")
raw, events = raw.resample(512, events=events, verbose="error")
raw.filter(l_freq=0.1, h_freq=30.0, verbose="error")

sampling_rate = raw.info["sfreq"]


# ---------------------------------------------------------------------
# 7. Find three consecutive stimulus-response-feedback cycles
# ---------------------------------------------------------------------

# Build a flat list of (sample, label) pairs for the three event types.
simple_events = []
for event in events:
    event_sample = event[0]
    event_code = int(event[2])
    if event_code in stimulus_codes:
        simple_events.append([event_sample, "Stimulus"])
    if event_code in response_codes:
        simple_events.append([event_sample, "Response"])
    if event_code in feedback_codes:
        simple_events.append([event_sample, "Feedback"])

target_sequence = [
    "Stimulus", "Response", "Feedback",
    "Stimulus", "Response", "Feedback",
    "Stimulus", "Response", "Feedback",
]

# Default window; the search below overwrites this when it finds a match.
three_cycle_start = simple_events[0][0] / sampling_rate
three_cycle_stop = three_cycle_start + 12.0

for start_index in range(len(simple_events) - len(target_sequence) + 1):
    labels = []
    for offset in range(len(target_sequence)):
        labels.append(simple_events[start_index + offset][1])
    if labels == target_sequence:
        three_cycle_start = simple_events[start_index][0] / sampling_rate - 0.5
        three_cycle_stop = simple_events[start_index + len(target_sequence) - 1][0] / sampling_rate + 0.5
        break


# ---------------------------------------------------------------------
# 8. Plot raw EEG — six figures (three windows × with / without events)
# ---------------------------------------------------------------------

# Each figure crops one time window, thins the samples for fast plotting,
# stacks channels vertically, and optionally overlays coloured event lines.


# --- full recording ---

raw_view = raw.copy().pick(biosemi64_channels[:32]).crop(tmin=0.0, tmax=raw.times[-1])
data_uv = raw_view.get_data() * 1e6
times = raw_view.times
decimation = max(1, int(np.ceil(len(times) / 7000)))
data_uv = data_uv[:, ::decimation]
times = times[::decimation]
spacing = max(20.0, float(data_uv.std()) * 6.0)

fig, ax = plt.subplots(figsize=(13, 7))
for ch in range(data_uv.shape[0]):
    ax.plot(times, data_uv[ch] + ch * spacing, color="#222222", linewidth=0.55)

saw_stimulus = False
saw_response = False
saw_feedback = False
for event in events:
    t = event[0] / sampling_rate
    code = int(event[2])
    if code in stimulus_codes:
        ax.axvline(t, color=stimulus_color, alpha=0.65, linewidth=0.9)
        saw_stimulus = True
    if code in response_codes:
        ax.axvline(t, color=response_color, alpha=0.65, linewidth=0.9)
        saw_response = True
    if code in feedback_codes:
        ax.axvline(t, color=feedback_color, alpha=0.65, linewidth=0.9)
        saw_feedback = True
legend_handles = []
if saw_stimulus:
    legend_handles.append(plt.Line2D([0], [0], color=stimulus_color, linewidth=2, label="Stimulus"))
if saw_response:
    legend_handles.append(plt.Line2D([0], [0], color=response_color, linewidth=2, label="Response"))
if saw_feedback:
    legend_handles.append(plt.Line2D([0], [0], color=feedback_color, linewidth=2, label="Feedback"))
ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=3)

ax.set_title("Raw EEG — full recording with event markers")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Channels (uV, offset)")
ax.set_yticks(np.arange(data_uv.shape[0]) * spacing)
ax.set_yticklabels(raw_view.ch_names, fontsize=7)
ax.grid(alpha=0.18)
fig.tight_layout()
fig.savefig("figures/01_raw_all_time_with_events.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 7))
for ch in range(data_uv.shape[0]):
    ax.plot(times, data_uv[ch] + ch * spacing, color="#222222", linewidth=0.55)
ax.set_title("Raw EEG — full recording")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Channels (uV, offset)")
ax.set_yticks(np.arange(data_uv.shape[0]) * spacing)
ax.set_yticklabels(raw_view.ch_names, fontsize=7)
ax.grid(alpha=0.18)
fig.tight_layout()
fig.savefig("figures/01_raw_all_time_no_events.png", dpi=150)
plt.close(fig)


# --- first half ---

half_stop = raw.times[-1] / 2.0
raw_view = raw.copy().pick(biosemi64_channels[:32]).crop(tmin=0.0, tmax=half_stop)
data_uv = raw_view.get_data() * 1e6
times = raw_view.times
decimation = max(1, int(np.ceil(len(times) / 7000)))
data_uv = data_uv[:, ::decimation]
times = times[::decimation]
spacing = max(20.0, float(data_uv.std()) * 6.0)

fig, ax = plt.subplots(figsize=(13, 7))
for ch in range(data_uv.shape[0]):
    ax.plot(times, data_uv[ch] + ch * spacing, color="#222222", linewidth=0.55)

saw_stimulus = False
saw_response = False
saw_feedback = False
for event in events:
    t = event[0] / sampling_rate
    code = int(event[2])
    if t <= half_stop:
        if code in stimulus_codes:
            ax.axvline(t, color=stimulus_color, alpha=0.65, linewidth=0.9)
            saw_stimulus = True
        if code in response_codes:
            ax.axvline(t, color=response_color, alpha=0.65, linewidth=0.9)
            saw_response = True
        if code in feedback_codes:
            ax.axvline(t, color=feedback_color, alpha=0.65, linewidth=0.9)
            saw_feedback = True
legend_handles = []
if saw_stimulus:
    legend_handles.append(plt.Line2D([0], [0], color=stimulus_color, linewidth=2, label="Stimulus"))
if saw_response:
    legend_handles.append(plt.Line2D([0], [0], color=response_color, linewidth=2, label="Response"))
if saw_feedback:
    legend_handles.append(plt.Line2D([0], [0], color=feedback_color, linewidth=2, label="Feedback"))
ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=3)

ax.set_title("Raw EEG — first half with event markers")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Channels (uV, offset)")
ax.set_yticks(np.arange(data_uv.shape[0]) * spacing)
ax.set_yticklabels(raw_view.ch_names, fontsize=7)
ax.grid(alpha=0.18)
fig.tight_layout()
fig.savefig("figures/02_raw_half_time_with_events.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 7))
for ch in range(data_uv.shape[0]):
    ax.plot(times, data_uv[ch] + ch * spacing, color="#222222", linewidth=0.55)
ax.set_title("Raw EEG — first half")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Channels (uV, offset)")
ax.set_yticks(np.arange(data_uv.shape[0]) * spacing)
ax.set_yticklabels(raw_view.ch_names, fontsize=7)
ax.grid(alpha=0.18)
fig.tight_layout()
fig.savefig("figures/02_raw_half_time_no_events.png", dpi=150)
plt.close(fig)


# --- three stimulus-response-feedback cycles ---

raw_view = raw.copy().pick(biosemi64_channels[:32]).crop(tmin=three_cycle_start, tmax=three_cycle_stop)
data_uv = raw_view.get_data() * 1e6
# raw_view.times resets to 0 after cropping; add the crop offset to
# keep the x-axis aligned with the absolute recording timeline.
times = raw_view.times + three_cycle_start
decimation = max(1, int(np.ceil(len(times) / 5000)))
data_uv = data_uv[:, ::decimation]
times = times[::decimation]
spacing = max(20.0, float(data_uv.std()) * 6.0)

fig, ax = plt.subplots(figsize=(13, 7))
for ch in range(data_uv.shape[0]):
    ax.plot(times, data_uv[ch] + ch * spacing, color="#222222", linewidth=0.55)

saw_stimulus = False
saw_response = False
saw_feedback = False
for event in events:
    t = event[0] / sampling_rate
    code = int(event[2])
    if three_cycle_start <= t <= three_cycle_stop:
        if code in stimulus_codes:
            ax.axvline(t, color=stimulus_color, alpha=0.65, linewidth=0.9)
            saw_stimulus = True
        if code in response_codes:
            ax.axvline(t, color=response_color, alpha=0.65, linewidth=0.9)
            saw_response = True
        if code in feedback_codes:
            ax.axvline(t, color=feedback_color, alpha=0.65, linewidth=0.9)
            saw_feedback = True
legend_handles = []
if saw_stimulus:
    legend_handles.append(plt.Line2D([0], [0], color=stimulus_color, linewidth=2, label="Stimulus"))
if saw_response:
    legend_handles.append(plt.Line2D([0], [0], color=response_color, linewidth=2, label="Response"))
if saw_feedback:
    legend_handles.append(plt.Line2D([0], [0], color=feedback_color, linewidth=2, label="Feedback"))
ax.legend(handles=legend_handles, loc="upper right", fontsize=8, ncol=3)

ax.set_title("Raw EEG — three stimulus-response-feedback cycles with event markers")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Channels (uV, offset)")
ax.set_yticks(np.arange(data_uv.shape[0]) * spacing)
ax.set_yticklabels(raw_view.ch_names, fontsize=7)
ax.grid(alpha=0.18)
fig.tight_layout()
fig.savefig("figures/03_raw_three_cycles_with_events.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 7))
for ch in range(data_uv.shape[0]):
    ax.plot(times, data_uv[ch] + ch * spacing, color="#222222", linewidth=0.55)
ax.set_title("Raw EEG — three stimulus-response-feedback cycles")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Channels (uV, offset)")
ax.set_yticks(np.arange(data_uv.shape[0]) * spacing)
ax.set_yticklabels(raw_view.ch_names, fontsize=7)
ax.grid(alpha=0.18)
fig.tight_layout()
fig.savefig("figures/03_raw_three_cycles_no_events.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------
# 9. Cut epochs around stimulus and feedback events
# ---------------------------------------------------------------------

stimulus_epochs = mne.Epochs(
    raw,
    events,
    event_id={"stimulus_20": 20, "stimulus_21": 21, "stimulus_22": 22, "stimulus_23": 23},
    tmin=-0.2,
    tmax=0.8,
    baseline=(None, 0.0),
    picks=biosemi64_channels,
    preload=True,
    verbose="error",
)

feedback_epochs = mne.Epochs(
    raw,
    events,
    event_id={"feedback_40": 40, "feedback_41": 41, "feedback_42": 42, "feedback_43": 43},
    tmin=-0.2,
    tmax=0.8,
    baseline=(None, 0.0),
    picks=biosemi64_channels,
    preload=True,
    verbose="error",
)

stimulus_evoked = stimulus_epochs.average()
feedback_evoked = feedback_epochs.average()


# ---------------------------------------------------------------------
# 10. Plot the stimulus- and feedback-aligned averages
# ---------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False, sharey=False)

stimulus_evoked.plot(axes=axes[0], spatial_colors=True, gfp=True, time_unit="s", show=False)
axes[0].set_title(f"Stimulus-aligned average (n={stimulus_evoked.nave})")

feedback_evoked.plot(axes=axes[1], spatial_colors=True, gfp=True, time_unit="s", show=False)
axes[1].set_title(f"Feedback-aligned average (n={feedback_evoked.nave})")

fig.suptitle("Evoked averages aligned to stimulus and feedback", y=0.995)
fig.subplots_adjust(top=0.90, hspace=0.55)
fig.savefig("figures/04_evoked_mne.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------
# 11. Plot the same two averages with scalp maps
# ---------------------------------------------------------------------

# Layout constants shared by both rows.
trace_left = 0.09
trace_width = 0.84
trace_height = 0.12
map_size = 0.065
map_gap = 0.045
topomap_times = [0.0, 0.1, 0.2, 0.3, 0.4]

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Event-aligned averages and scalp maps", y=0.98)


# --- stimulus row ---

trace_bottom = 0.57
trace_ax = fig.add_axes([trace_left, trace_bottom, trace_width, trace_height])
trace_ax.plot(stimulus_evoked.times, stimulus_evoked.data.T * 1e6, linewidth=0.6)
trace_ax.axvline(0.0, color="#444444", linewidth=1.0)
trace_ax.set_title(f"Stimulus-aligned average (n={stimulus_evoked.nave})", pad=8)
trace_ax.set_xlabel("Time (s)")
trace_ax.set_ylabel("uV")
trace_ax.grid(alpha=0.2)

y_top = trace_ax.get_ylim()[1]
row_tmin = stimulus_evoked.times[0]
row_tmax = stimulus_evoked.times[-1]

for map_time in topomap_times:
    map_fraction = (map_time - row_tmin) / (row_tmax - row_tmin)
    map_center = trace_left + trace_width * map_fraction
    map_ax = fig.add_axes([map_center - map_size / 2.0, trace_bottom + trace_height + map_gap, map_size, map_size])
    stimulus_evoked.plot_topomap(times=[map_time], axes=map_ax, colorbar=False, show=False, time_unit="s")
    map_ax.set_title(f"{map_time:.1f} s")
    trace_ax.axvline(map_time, color="#aaaaaa", linewidth=1.0)
    stem = ConnectionPatch(
        xyA=(0.5, 0.0), coordsA=map_ax.transAxes,
        xyB=(map_time, y_top), coordsB=trace_ax.transData,
        color="#aaaaaa", linewidth=1.0,
    )
    fig.add_artist(stem)


# --- feedback row ---

trace_bottom = 0.18
trace_ax = fig.add_axes([trace_left, trace_bottom, trace_width, trace_height])
trace_ax.plot(feedback_evoked.times, feedback_evoked.data.T * 1e6, linewidth=0.6)
trace_ax.axvline(0.0, color="#444444", linewidth=1.0)
trace_ax.set_title(f"Feedback-aligned average (n={feedback_evoked.nave})", pad=8)
trace_ax.set_xlabel("Time (s)")
trace_ax.set_ylabel("uV")
trace_ax.grid(alpha=0.2)

y_top = trace_ax.get_ylim()[1]
row_tmin = feedback_evoked.times[0]
row_tmax = feedback_evoked.times[-1]

for map_time in topomap_times:
    map_fraction = (map_time - row_tmin) / (row_tmax - row_tmin)
    map_center = trace_left + trace_width * map_fraction
    map_ax = fig.add_axes([map_center - map_size / 2.0, trace_bottom + trace_height + map_gap, map_size, map_size])
    feedback_evoked.plot_topomap(times=[map_time], axes=map_ax, colorbar=False, show=False, time_unit="s")
    map_ax.set_title(f"{map_time:.1f} s")
    trace_ax.axvline(map_time, color="#aaaaaa", linewidth=1.0)
    stem = ConnectionPatch(
        xyA=(0.5, 0.0), coordsA=map_ax.transAxes,
        xyB=(map_time, y_top), coordsB=trace_ax.transData,
        color="#aaaaaa", linewidth=1.0,
    )
    fig.add_artist(stem)

fig.savefig("figures/05_evoked_joint_mne.png", dpi=150)
plt.close(fig)


# ---------------------------------------------------------------------
# 12. Plot the reaction time distribution
# ---------------------------------------------------------------------

reaction_times_ms = behaviour["rt"]
reaction_times_seconds = reaction_times_ms / 1000.0

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(reaction_times_seconds, bins=30, color="#4c78a8", edgecolor="white")
ax.set_xlabel("Reaction time (s)")
ax.set_ylabel("Number of trials")
ax.set_title("Reaction time distribution")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig("figures/06_reaction_time_histogram.png", dpi=150)
plt.close(fig)
