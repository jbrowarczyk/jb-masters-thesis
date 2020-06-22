EEG_CHANNELS = ['AF3','F3','F7','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
N_CHANNELS = len(EEG_CHANNELS)
FS = 128                         # samples per second
FRAME_DURATION = 1               # seconds
N_SAMPLES = FS * FRAME_DURATION  # samples per one frame
N_OVERLAP = 64                   # number of overlapping samples in consecutive frames
TRAIN_VERBOSE = False            # show training log during SVM and neural network training