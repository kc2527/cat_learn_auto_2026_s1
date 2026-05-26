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

# NOTE: pre sets
# set montage
montage = mne.channels.make_standard_montage('biosemi64')

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

# NOTE: prep for bad channels 
# read in .csv with notes on bad channels
bad_channs = pd.read_csv('/Users/kayla/Desktop/KC/university/2026/projects/cat_learn_auto_2026_s1/eeg/meta_data/eeg_bad_channs.csv')

# remove empty columns
bad_channs = bad_channs.dropna(axis=1, how='all')

# ensure that any trailing spaces are no longer there
bad_channs["participant"] = bad_channs["participant"].astype(str).str.strip()

# TODO: not including perceptual baseline in this for the moment
bad_channs = bad_channs[bad_channs['session_num'] !=  0].copy()

# create empty column in bad_channs
bad_channs["list_bads"] = None

# creates list of the bad channels from bad_channs in line with
tmp = []
for x in bad_channs["bad_channels"]:
    if pd.isna(x) or str(x).strip() == "":
        tmp.append([])
    else:
        tmp.append([c.strip() for c in str(x).split(",")])
bad_channs['list_bads'] = tmp

# creates a dictionary of the bad channels
bad_lookup = {}
for _, row in bad_channs.iterrows():
    key = (row["participant"], int(row["session_num"]))
    bad_lookup[key] = row["list_bads"]

# NOTE: create empty list to log bad channels and exluded ICA across participants
log_processes = []

# NOTE: set in and out paths
in_root_dir = '/Users/kayla/Desktop/KC/university/2026/projects/cat_learn_auto_2026_s1/eeg/simulated_data'

# output root
out_root_dir = "/Users/kayla/Desktop/KC/university/2026/projects/cat_learn_auto_2026_s1/eeg/cleaned_simulated_data"

# create empty list for path names
file_paths = []

# append list with path names for every file
for fd in os.listdir(in_root_dir):
    in_root_dir_fd = os.path.join(in_root_dir, fd)
    if os.path.isdir(in_root_dir_fd):
        for fs in os.listdir(in_root_dir_fd):
            f_full_path = os.path.join(in_root_dir_fd, fs)
            if os.path.isfile(f_full_path) and fs.endswith(".bdf"):
                file_paths.append(f_full_path)

# subject IDs to skip
exclude_participants = {}

# NOTE: start of pre-processing loop
for fpath in file_paths:
    fname = os.path.basename(fpath)
    stem = os.path.splitext(fname)[0]
    participant, task, session_str = stem.split("_")
    session = int(session_str)
    
    if participant in exclude_participants:
        continue

    # read file in
    raw = mne.io.read_raw_bdf(fpath, preload=True)

    # find events
    events = mne.find_events(raw, stim_channel='Status')

    # down sample data 
    raw, events = raw.resample(512, events=events, verbose="error")

    # NOTE: label channels
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

    # set montage to biosemi64 (from up above)
    raw.set_montage(montage, on_missing="ignore")

    # NOTE: highpass filter
    raw.filter(l_freq=0.1, h_freq=None)

    # NOTE: find bridged channels and save to output file
    raw_bridge = raw.copy().pick("eeg")
    bridged_idx, ed_matrix = mne.preprocessing.compute_bridged_electrodes(raw_bridge)
    bridged_pairs = [(raw_bridge.ch_names[i], raw_bridge.ch_names[j]) for i, j in bridged_idx]

    # NOTE: handle bad channels and interpolate
    raw.info["bads"] = []
    convert = dict(zip(raw_scalp_channels, biosemi64_channels))

    # know that B31 was a bad channel in P268_eeg_5 session
    known_bads = bad_lookup.get((participant, session), [])
    convert_bads = [convert[ch] for ch in known_bads if ch in convert]

    # different methods are used to automate finding bads -- here is Jordans + neurokit
    ch_names = biosemi64_channels
    bad_index = nk.eeg_badchannels(raw, bad_threshold=0.5, distance_threshold=0.99, show=False)
    bad_idx = bad_index[0]
    temp = raw.get_data()
    temp = temp[0:64, :]
    temp_SD = np.std(temp, 1)                               # sd of each channel
    mean_SD = np.mean(temp_SD)                              # find the mean of the sds
    sd_SD = np.std(temp_SD)                                 # find the sd of the sds
    sd_idx = np.where(np.abs(temp_SD)>mean_SD+sd_SD*2)[0]   # and find a cutoff
    bad = [ch_names[i] for i in sd_idx]
    bads = np.unique(np.concatenate((bad, bad_idx)))
    raw.info['bads'] = bads.tolist()

    # add known bads to bads from calculations
    raw.info["bads"] = list(set(raw.info["bads"]).union(convert_bads))

    # creating copy of bads for log before they get wiped by interpolate
    log_bads = raw.info['bads'].copy()

    # interpolate
    raw.interpolate_bads(reset_bads=True)
    # num_bad_ch.append(len(bads))

    # NOTE: ICA + filtering
    filt_raw = raw.copy().filter(l_freq=1, h_freq = 20)

    # run ICA
    ica = mne.preprocessing.ICA(max_iter="auto") #Pick 20 to speed up
    ica.fit(filt_raw)               # Fit
    # ica.plot_components()         # Plot heads
    # ica.plot_sources(raw)         # Plot timelines
    ica.exclude = []                # Could set these manually
    eog_indices, eog_scores = ica.find_bads_eog(filt_raw)
    ica.exclude = eog_indices

    # now remove components
    ica.apply(raw)

    # NOTE: log bad channels, ICA components, and bridged channels for this file
    # and append to list for file log
    log_processes.append({
        "participant": participant,
        "session": session,
        "task": task,
        "file_name": fname,
        "manual_bads_raw": ",".join(known_bads),
        "manual_bads_montage": ",".join(convert_bads),
        "calculated_bads": ",".join(bads.tolist()),
        "final_bads_used": ",".join(log_bads),
        "ica_components_excluded": ",".join(map(str, eog_indices)),
        "ica_component_count": len(eog_indices),
        "bridged_pairs": ";".join([f"{a}-{b}" for a, b in bridged_pairs]),
        "n_bridged_pairs": len(bridged_pairs)
    })

    # get rid of EOG
    raw.pick(picks = ['eeg'])

    # broad filtering -- 0.1 - 30
    raw.filter(l_freq=0.5, h_freq = 45)

    # filtering out line noise
    freqs_powerline = (50, 100, 150, 200)
    raw = raw.notch_filter(freqs=freqs_powerline)

    # use the average of all channels as reference
    raw.set_eeg_reference(ref_channels='average')

    # psd
    # raw.compute_psd(fmin = 0.5 , fmax = 40).plot()

    # NOTE: epoching, detrending, and baseline correcting
    # baseline correct is automatic with epoching
    # TODO: for test run, making event_id=None, change back later
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=1, baseline=(-0.2, 0),
                        event_id=None, detrend=0, reject=None,
                        reject_by_annotation=None,  preload=True)

    # NOTE: saving cleaned file
    # keep session folder structure (eeg_1, eeg_2, ...)
    session_folder = os.path.basename(os.path.dirname(fpath))   # e.g., "eeg_1"
    out_dir = os.path.join(out_root_dir, session_folder)
    os.makedirs(out_dir, exist_ok=True)

    # save cleaned continuous data
    out_fname = f"{participant}_{task}_{session}_clean_raw.fif"
    out_path = os.path.join(out_dir, out_fname)
    raw.save(out_path, overwrite=True)

    # save epochs
    epo_out = os.path.join(out_dir, f"{participant}_{task}_{session}_clean_epo.fif")
    epochs.save(epo_out, overwrite=True)
        
# --- after processing ---
# create csv for all bad channels found across all participants
log_processes = pd.DataFrame(log_processes)
log_processes.to_csv(
    "/Users/kayla/Desktop/KC/university/2026/projects/cat_learn_auto_2026_s1/eeg/meta_data/bad_channels_run_log.csv",
    index=False
)

# stim_A_train = epochs["STIM_ONSET_A_TRAIN"]
# stim_B_train = epochs["STIM_ONSET_B_TRAIN"]
# stim_A_probe = epochs["STIM_ONSET_A_PROBE"]
# stim_B_probe = epochs["STIM_ONSET_B_PROBE"]
# 
# resp_A_train = epochs["RESP_A_TRAIN"]
# resp_B_train = epochs["RESP_B_TRAIN"]
# resp_A_probe = epochs["RESP_A_PROBE"]
# resp_B_probe = epochs["RESP_B_PROBE"]
# 
# fb_cor_train = epochs["FB_COR_TRAIN"]
# fb_inc_train = epochs["FB_INC_TRAIN"]
# fb_cor_probe = epochs["FB_COR_PROBE"]
# fb_inc_probe = epochs["FB_INC_PROBE"]
