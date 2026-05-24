# eeg analysis steps
* I was a dumbass and forgot about the erp book so here are the steps I would do with respect to that

1. load in the data and downsample down to 512? (because ours is sampled at 4096)
- make sure events still align and follow down sampling
2. find events
3. label channels 
4. run a high pass filter - unsure of what this would be at though (0.01 or 0.1)
5. handle bad channels and interpolate
6. artefact detection
- ICA before epoching for eye blinks 
- broad filtering and the line noise 

and then epoch - d trend (does the same thing as a high pass filter) and
baseline correct 

dont use rejection criteria when epoching -- save epoch data, then load it somewhere else and then reject epoch (based on criteria)


9. artefact rejection (so I imagine this includes ICA, eog, and line noise)
7. re-reference (either to mastoids or average reference)
8. epoch data + baseline correction
10. save cleaned and epoched files -- this last step I am unsure about. 

Something I am unsure about though is saving cleaned, processed files which you can then pull into an analysis pipeline. Should those cleaned files already be epoched with baseline correction and artefact rejection? Or would you do artefact rejection prior to epoching, save the processed file, then epoch and baseline correct in the analysis? 