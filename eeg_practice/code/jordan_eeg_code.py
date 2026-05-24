import mne
import numpy as np
import glob
import scipy
import matplotlib.pyplot as plt
import neurokit2 as nk #pitp install this package
import pandas as pd
import pingouin as pg
import scipy
from mne.stats import permutation_cluster_1samp_test
import mne_icalabel
# from JW_packages import cleaning as cl
# %matplotlib qt

ica_count = [] #number of ICA components
name_store = [] #names
check = [] #checking event codes
num_bad_ch = [] #Number of bad channels

files = glob.glob('C:/Users/jorda/Documents/Students/*bdf')
montage= mne.channels.make_standard_montage('biosemi64')

#%%
for file in files: #Subject 5 has bad data, might need to adjust

    #File name
    name = file.split('/')[-1].split('.')[0].split('\\')[1]

    #Import
    raw = mne.io.read_raw_bdf(file, preload = True)
    raw.set_channel_types({'EXG1': 'eog', 'EXG2': 'eog', 'EXG3': 'bio', 'EXG4': 'bio'})

    #Get events and timing
    events = mne.events_from_annotations(raw)[0]
    raw.pick(picks = ['eeg', 'eog'])

    #Get channel names
    ch_names = raw.ch_names
    dictionary_name_change = dict(zip(ch_names[0:65], montage.ch_names))
    raw.rename_channels(dictionary_name_change)
    raw.set_montage('biosemi64', on_missing = 'ignore')
    ch_names = raw.ch_names

    #Keepting record of the events
    check.append(np.unique(events[:,2]))

    #And names
    name_store.append(name)

    #Find bad channels
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

    #Now we can mark this as bad
    raw.info['bads'] = bads.tolist()
    raw.interpolate_bads(reset_bads=True)
    num_bad_ch.append(len(bads))

    #ICA
    filt_raw = raw.copy().filter(l_freq=1, h_freq = 20)

    #Run ICA
    ica = mne.preprocessing.ICA(max_iter="auto") #Pick 20 to speed up
    ica.fit(filt_raw) #Fit
    #ica.plot_components() #Plot heads
    #ica.plot_sources(raw) #Plot timelines
    ica.exclude = [] #Could set these manually
    eog_indices, eog_scores = ica.find_bads_eog(filt_raw)
    ica.exclude = eog_indices
    ica_count.append(len(eog_indices))

    #Now remove these components
    ica.apply(raw)

    #Get rid of EOG
    raw.pick(picks = ['eeg'])
    raw.filter(l_freq=0.5, h_freq = 45) #Broad is 0.1 to 30

    freqs_powerline = (50, 100, 150, 200)
    raw = raw.notch_filter(freqs=freqs_powerline)
    raw.set_eeg_reference()

    #psd
    raw.compute_psd(fmin = .5 , fmax = 40).plot()
    #Epoch the data
