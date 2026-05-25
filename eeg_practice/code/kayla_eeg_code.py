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

# NOTE: load and downsample data
# prep for file
dir_data_eeg = '/Users/kayla/Desktop/KC/university/2026/projects/cat_learn_auto_2026_s1/eeg_practice/eeg_data'
f = os.path.join(dir_data_eeg, 'P268_eeg_5.bdf')

# event dict
event_dict = {

        # -------------------- Experiment structure --------------------
        "EXP_START": 10,
        "ITI_ONSET": 11,
        "EXP_END": 15,

        # -------------------- Stimulus onset --------------------
        # Training trials
        "STIM_ONSET_A_TRAIN": 20,
        "STIM_ONSET_B_TRAIN": 21,

        # Probe trials
        "STIM_ONSET_A_PROBE": 22,
        "STIM_ONSET_B_PROBE": 23,

        # -------------------- Responses --------------------
        # Training trials
        "RESP_A_TRAIN": 30,
        "RESP_B_TRAIN": 31,

        # Probe trials
        "RESP_A_PROBE": 32,
        "RESP_B_PROBE": 33,

        # -------------------- Feedback --------------------
        # Training trials
        "FB_COR_TRAIN": 40,
        "FB_INC_TRAIN": 41,

        # Probe trials
        "FB_COR_PROBE": 42,
        "FB_INC_PROBE": 43,
}

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

# NOTE: label channels
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

# set channels for eog
raw.set_channel_types({"EXG1": "eog", "EXG2": "eog"}, on_unit_change="ignore")

# rename from A/B 1-32 to meaningful electrode names
rename_map = {}
for i in range(len(raw_scalp_channels)):
    rename_map[raw_scalp_channels[i]] = biosemi64_channels[i]
raw.rename_channels(rename_map)

# set montage
raw.set_montage(mne.channels.make_standard_montage("biosemi64"), on_missing="ignore")

# NOTE: highpass filter
raw.filter(l_freq=0.1, h_freq=None)

# NOTE: handle bad channels and interpolate
raw.info["bads"] = []
convert = dict(zip(raw_scalp_channels, biosemi64_channels))

# know that B31 was a bad channel in P268_eeg_5 session
known_bads = ['B31']
convert_bads = [convert[ch] for ch in known_bads if ch in convert]

# different methods are used to automate finding bads -- here is Jordans
ch_names = biosemi64_channels
bad_index = nk.eeg_badchannels(raw, bad_threshold=0.5, distance_threshold=0.99, show=False)
bad_idx = bad_index[0]
temp = raw.get_data()
temp = temp[0:62, :]
temp_SD = np.std(temp, 1)                               # sd of each channel
mean_SD = np.mean(temp_SD)                              # find the mean of the sds
sd_SD = np.std(temp_SD)                                 # find the sd of the sds
sd_idx = np.where(np.abs(temp_SD)>mean_SD+sd_SD*2)[0]   # and find a cutoff
bad = [ch_names[i] for i in sd_idx]
bads = np.unique(np.concatenate((bad, bad_idx)))
raw.info['bads'] = bads.tolist()
print(raw.info['bads'])

# add known bads to bads from calculations
raw.info["bads"] = list(set(raw.info["bads"]).union(convert_bads))

# interpolate
raw.interpolate_bads(reset_bads=True)
# num_bad_ch.append(len(bads))

# NOTE: ICA + filtering
filt_raw = raw.copy().filter(l_freq=1, h_freq = 20)

#Run ICA
ica = mne.preprocessing.ICA(max_iter="auto") #Pick 20 to speed up
ica.fit(filt_raw) #Fit
# ica.plot_components()         # Plot heads
# ica.plot_sources(raw)         # Plot timelines
ica.exclude = []                # Could set these manually
eog_indices, eog_scores = ica.find_bads_eog(filt_raw)
ica.exclude = eog_indices
# ica_count.append(len(eog_indices))

# now remove components
ica.apply(raw)

# get rid of EOG
raw.pick(picks = ['eeg'])

# use the average of all channels as reference
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels="average")
raw_avg_ref.plot()

# broad filtering -- 0.1 - 30
raw.filter(l_freq=0.5, h_freq = 45) 

# filtering out line noise
freqs_powerline = (50, 100, 150, 200)
raw = raw.notch_filter(freqs=freqs_powerline)
raw.set_eeg_reference()

# psd
raw.compute_psd(fmin = 0.5 , fmax = 40).plot()

# sanity check plot
raw.plot(
    picks="eeg",
    duration=8,        # short window
    n_channels=32,
    remove_dc=True,
    highpass=1.0,      # display-only
    lowpass=40.0,      # display-only
    scalings=dict(eeg=40e-6),
    decim=2
)

# NOTE: epoching, detrending, and baseline correcting
# baseline correct is automatic with epoching
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=1, baseline=(-0.2, 0),
                    event_id=event_dict, detrend=0, reject=None,
                    reject_by_annotation=None,  preload=True)

# stim onset A train = 20
mne.Epochs(raw, events, event_id={"20": 20, "21": 21}, tmin=-0.2, tmax=1.0,
           baseline=(-0.2, 0), preload=True).average().plot()

stim_A_train = epochs["STIM_ONSET_A_TRAIN"]
stim_B_train = epochs["STIM_ONSET_B_TRAIN"]
stim_A_probe = epochs["STIM_ONSET_A_PROBE"]
stim_B_probe = epochs["STIM_ONSET_B_PROBE"]

resp_A_train = epochs["RESP_A_TRAIN"]
resp_B_train = epochs["RESP_B_TRAIN"]
resp_A_probe = epochs["RESP_A_PROBE"]
resp_B_probe = epochs["RESP_B_PROBE"]

fb_cor_train = epochs["FB_COR_TRAIN"]
fb_inc_train = epochs["FB_INC_TRAIN"]
fb_cor_probe = epochs["FB_COR_PROBE"]
fb_inc_probe = epochs["FB_INC_PROBE"]
