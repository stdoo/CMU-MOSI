from constants import DATA_PATH
import os
import re
import pickle
import numpy as np
from mmsdk import mmdatasdk as md
from subprocess import check_call

# create folders for storing the data
if not os.path.exists(DATA_PATH):
    check_call(' '.join(['mkdir', '-p', DATA_PATH]), shell=True)
data_files = os.listdir(DATA_PATH)
print('\n'.join(data_files))
DATASET = md.cmu_mosi

try:
    md.mmdataset(DATASET.highlevel, DATA_PATH)
except RuntimeError:
    print("High-level features have been downloaded previously.")

try:
    md.mmdataset(DATASET.raw, DATA_PATH)
except RuntimeError:
    print("Raw data have been downloaded previously.")
    
try:
    md.mmdataset(DATASET.labels, DATA_PATH)
except RuntimeError:
    print("Labels have been downloaded previously.")


# list the directory contents... let's see what features there are
data_files = os.listdir(DATA_PATH)
print('\n'.join(data_files))

visual_field = 'CMU_MOSI_VisualFacet_4.1'
acoustic_field = 'CMU_MOSI_COVAREP'
text_field = 'CMU_MOSI_ModifiedTimestampedWords'

features = [
    text_field, 
    visual_field, 
    acoustic_field
]

recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
dataset = md.mmdataset(recipe)


# we define a simple averaging function that does not depend on intervals
def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features


# first we align to words with averaging, collapse_function receives a list of functions
dataset.align(text_field, collapse_functions=[avg])

label_field = 'CMU_MOSI_Opinion_Labels'

label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
dataset.add_computational_sequences(label_recipe, destination=None)
dataset.align(label_field)

train_split = DATASET.standard_folds.standard_train_fold
dev_split = DATASET.standard_folds.standard_valid_fold
test_split = DATASET.standard_folds.standard_test_fold

# inspect the splits: they only contain video IDs
print(test_split)

# we can see they are in the format of 'video_id[segment_no]', but the splits was specified with video_id only
# we need to use regex or something to match the video IDs...

# a sentinel epsilon for safe division, without it we will replace illegal values with a constant
EPS = 0

# place holders for the final train/dev/test dataset
train = []
dev = []
test = []

# define a regular expression to extract the video ID out of the keys
pattern = re.compile('(.*)\[.*\]')
num_drop = 0  # a counter to count how many data points went into some processing issues

for segment in dataset[label_field].keys():
    
    # get the video ID and the features out of the aligned dataset
    vid = re.search(pattern, segment).group(1)
    label = dataset[label_field][segment]['features']
    _words = dataset[text_field][segment]['features']
    _visual = dataset[visual_field][segment]['features']
    _acoustic = dataset[acoustic_field][segment]['features']

    # if the sequences are not same length after alignment, there must be some problem with some modalities
    # we should drop it or inspect the data again
    if not _words.shape[0] == _visual.shape[0] == _acoustic.shape[0]:
        print(f"Encountered datapoint {vid} with text shape {_words.shape}, visual shape {_visual.shape}, acoustic shape {_acoustic.shape}")
        num_drop += 1
        continue

    # remove nan values
    label = np.nan_to_num(label)
    _visual = np.nan_to_num(_visual)
    _acoustic = np.nan_to_num(_acoustic)

    # remove speech pause tokens - this is in general helpful
    # we should remove speech pauses and corresponding visual/acoustic features together
    # otherwise modalities would no longer be aligned
    words = []
    visual = []
    acoustic = []
    for i, word in enumerate(_words):
        if word[0] != b'sp':
            words.append(word[0].decode('utf-8')) # SDK stores strings as bytes, decode into strings here
            visual.append(_visual[i, :])
            acoustic.append(_acoustic[i, :])

    words = ' '.join(words)
    visual = np.asarray(visual)
    acoustic = np.asarray(acoustic)

    # z-normalization per instance and remove nan/infs
    visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + np.std(visual, axis=0, keepdims=True)))
    acoustic = np.nan_to_num((acoustic - acoustic.mean(0, keepdims=True)) / (EPS + np.std(acoustic, axis=0, keepdims=True)))

    if vid in train_split:
        train.append(((words, visual, acoustic), label, segment))
    elif vid in dev_split:
        dev.append(((words, visual, acoustic), label, segment))
    elif vid in test_split:
        test.append(((words, visual, acoustic), label, segment))
    else:
        print(f"Found video that doesn't belong to any splits: {vid}")

print(f"Total number of {num_drop} datapoints have been dropped.")

storage_files = ['train', 'dev', 'test']
storage_values = [train, dev, test]

for i in range(len(storage_files)):
    pth = os.path.join(DATA_PATH, storage_files[i])
    if not os.path.isfile(pth):
        with open(pth, 'wb') as f:
            pickle.dump(storage_values[i], f)

