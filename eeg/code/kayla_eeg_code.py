import mne
import numpy as np
import os
import glob
import scipy
import matplotlib.pyplot as plt
import neurokit2 as nk 
import pandas as pd
import pingouin as pg
import scipy
from mne.stats import permutation_cluster_1samp_test
import mne_icalabel

# prep for file
dir_data_eeg = '/Users/kayla/Desktop/KC/university/2026/projects/cat_learn_auto_2026_s1/eeg/eeg_data'
f = os.path.join(dir_data_eeg, 'P268_eeg_5.bdf')

# NOTE: uncomment for event alignment checks
# raw = mne.io.read_raw_bdf(f, preload=True)
# events = mne.find_events(raw, stim_channel="Status")
# 
# sf_old = raw.info["sfreq"]          # store BEFORE resampling
# events_orig = events.copy()         # keep original event samples
# 
# raw_rs, events_rs = raw.resample(512, events=events_orig, verbose="error")
# sf_new = raw_rs.info["sfreq"]
# 
# t_old = events_orig[:, 0] / sf_old
# t_new = events_rs[:, 0] / sf_new
# dt_ms = (t_new - t_old) * 1000
# 
# print("count old/new:", len(events_orig), len(events_rs))
# print("max abs shift (ms):", np.max(np.abs(dt_ms)))
# print("mean abs shift (ms):", np.mean(np.abs(dt_ms)))

# read file in
raw = mne.io.read_raw_bdf(f, preload=True)

# find events
events = mne.find_events(raw, stim_channel='Status')

# down sample data 
raw, events = raw.resample(512, events=events, verbose="error")

# labelling channels
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

# label aux channels as misc
aux_channel_types = {}
for channel_name in raw.ch_names:
    if channel_name != "Status" and channel_name not in raw_scalp_channels:
        aux_channel_types[channel_name] = "misc"
raw.set_channel_types(aux_channel_types, on_unit_change="ignore")

# label scalp electrodes as eeg
scalp_channel_types = {}
for channel_name in raw_scalp_channels:
    scalp_channel_types[channel_name] = "eeg"
raw.set_channel_types(scalp_channel_types)

# set channels for eog and mastoids
raw.set_channel_types({"EXG1": "eog", "EXG2": "eog"}, on_unit_change="ignore")
raw.set_channel_types({"EXG3": "eeg", "EXG4": "eeg"}, on_unit_change="ignore")

# rename from A/B 1-32 to meaningful electrode names
rename_map = {}
for i in range(len(raw_scalp_channels)):
    rename_map[raw_scalp_channels[i]] = biosemi64_channels[i]
raw.rename_channels(rename_map)

# set montage
raw.set_montage(mne.channels.make_standard_montage("biosemi64"), on_missing="ignore")

# mne tutorial -- setting the eeg reference
# mastoids
raw.set_eeg_reference(ref_channels=['M1', 'M2'])

# add new reference channel (all zero)
raw_new_ref = mne.add_reference_channels(raw, ref_channels=[""])

# use the average of all channels as reference
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels="average")
raw_avg_ref.plot()

# creating average reference as a projector
raw.set_eeg_reference("average", projection=True)
print(raw.info["projs"])

#  mne tutorial -- artefact detection
raw.plot(duration=30, 
         n_channels=64,
         highpass=1.0,
         lowpass=40.0)

# psd plot
fig = raw.compute_psd(tmax=np.inf, fmax=250).plot(
    average=True, amplitude=False, picks=biosemi64_channels, exclude="bads"
)
plt.show()

# eog plots
eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
eog_epochs.plot_image(combine="mean")
eog_epochs.average().drop_channels(['EXG3', 'EXG4']).plot_joint()

# mne tutorial -- handling bad channels
# NOTE: for future, want to keep it to A/B system because those are the notes that we 
raw.info["bads"] = []

raw_qc = raw.copy()#.filter(1.0, 40.0, picks="eeg")
events_qc = mne.find_events(raw_qc)
epochs = mne.Epochs(raw_qc, events=events_qc)['20', '21'].average().plot()

# different methods are used to automate finding bads -- here is Jordans
ch_names = biosemi64_channels
bad_index = nk.eeg_badchannels(raw, bad_threshold=0.5, distance_threshold=0.99, show=False)
bad_idx = bad_index[0]
temp = raw.get_data()
temp = temp[0:62, :]
temp_SD = np.std(temp, 1) #sd of each channel
mean_SD = np.mean(temp_SD)# Find the mean of the sds
sd_SD = np.std(temp_SD) #find the sd of the sds
sd_idx = np.where(np.abs(temp_SD)>mean_SD+sd_SD*2)[0] #and find a cutoff
bad = [ch_names[i] for i in sd_idx]
bads = np.unique(np.concatenate((bad, bad_idx)))
raw.info['bads'] = bads.tolist()

# know that B31 was a bad channel in P268_eeg_5 session
raw.info["bads"].append('PO4')

eeg_data = raw.copy().pick(picks="eeg")
eeg_data_interp = eeg_data.copy().interpolate_bads(reset_bads=False)

for title, data in zip(["orig.", "interp."], [eeg_data, eeg_data_interp]):
    with mne.viz.use_browser_backend("matplotlib"):
        fig = data.plot(butterfly=True, color="#00000022", bad_color="r")
    fig.subplots_adjust(top=0.9)
    fig.suptitle(title, size="xx-large", weight="bold")


tmp = eeg_data.copy()
tmp.info["bads"] = [ch for ch in tmp.info["bads"] if ch not in ["EXG3", "EXG4", "M1", "M2"]]
eeg_data_interp = tmp.interpolate_bads(reset_bads=False)

# 1) Interpolate only true scalp channels from your montage names
scalp64 = [
    "Fp1","AF7","AF3","F1","F3","F5","F7","FT7","FC5","FC3","FC1","C1","C3","C5","T7","TP7",
    "CP5","CP3","CP1","P1","P3","P5","P7","P9","PO7","PO3","O1","Iz","Oz","POz","Pz","CPz",
    "Fpz","Fp2","AF8","AF4","AFz","Fz","F2","F4","F6","F8","FT8","FC6","FC4","FC2","FCz","Cz",
    "C2","C4","C6","T8","TP8","CP6","CP4","CP2","P2","P4","P6","P8","P10","PO8","PO4","O2"
]
tmp.pick([ch for ch in scalp64 if ch in tmp.ch_names])

# 2) Keep only bads that are still present after pick
tmp.info["bads"] = [ch for ch in tmp.info["bads"] if ch in tmp.ch_names]

# 3) Check duplicate coordinates
coords = np.array([tmp.info["chs"][tmp.ch_names.index(ch)]["loc"][:3] for ch in tmp.ch_names])
dupes = []
for i in range(len(coords)):
    for j in range(i+1, len(coords)):
        if np.allclose(coords[i], coords[j], atol=1e-12):
            dupes.append((tmp.ch_names[i], tmp.ch_names[j]))
print("duplicate coord pairs:", dupes[:20], "count=", len(dupes))

# 4) Retry interpolation
tmp_interp = tmp.interpolate_bads(reset_bads=False)

#### ^^^ AI help with that, don't know how it fixed the problem
#### but it seemed linked to channel labelling so will have to keep a better
#### track of that in the real deal

# mne tutorial -- repeairing artefacts with ICA
