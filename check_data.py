import os
import pickle
import torchtext

from constants import DATA_PATH

from mmsdk import mmdatasdk as md

# all the attributes of mmdatasdk
computational_sequence = md.computational_sequence  # class computational_sequence
mmdataset = md.mmdataset  # class mmdataset
cmu_mosei = md.cmu_mosei  # module cmu_mosei
cmu_mosi = md.cmu_mosi  # module cmu_mosi
pom = md.pom  # module pom

# all the attributes of cmu_mosei
standard_folds_1 = cmu_mosei.standard_folds
raw_1 = cmu_mosei.raw
highlevel_1 = cmu_mosei.highlevel
labels_1 = cmu_mosei.labels
extra = cmu_mosei.extra

# all the attributes of cmu_mosi
standard_folds_2 = cmu_mosi.standard_folds
raw_2 = cmu_mosi.raw
highlevel_2 = cmu_mosi.highlevel
labels_2 = cmu_mosi.labels

# all the attributes of pom
standard_folds_3 = pom.standard_folds
raw_3 = pom.raw
highlevel_3 = pom.highlevel
labels_3 = pom.labels

# all the attributes of standard_folds
standard_train_fold = standard_folds_1.standard_train_fold
standard_test_fold = standard_folds_1.standard_test_fold
standard_valid_fold = standard_folds_1.standard_valid_fold

# create a dataset (sequences already exist)
# 1. given the path of the directory, using all the sequences in it
print(os.listdir(DATA_PATH))
recipe_1 = DATA_PATH
dataset_1 = mmdataset(recipe_1)

# 2. given path to the sequences, using sequences in recipe
visual_field = 'CMU_MOSI_VisualFacet_4.1'
acoustic_field = 'CMU_MOSI_COVAREP'
text_field = 'CMU_MOSI_ModifiedTimestampedWords'
features = [
    text_field,
    visual_field,
    acoustic_field
]
recipe_2 = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
dataset_2 = mmdataset(recipe_2)

# 3. if sequences not exist, the value of the recipe should be URL， just like raw/highlevel/label
# recipe = highlevel_2
# dataset = mmdataset(recipe, destination='directory/to/keep/sequences')

# details in the dataset
print(list(dataset_2.keys()))  # seqs
print("=" * 80)

print(list(dataset_2[visual_field].keys())[:10])  # vid(video id) in visual seq
print("=" * 80)

some_id = list(dataset_2[visual_field].keys())[15]  # get some vid
print(list(dataset_2[visual_field][some_id].keys()))  # each vid has intervals and features
print("=" * 80)

print(list(dataset_2[visual_field][some_id]['intervals'].shape))
print("=" * 80)

print(list(dataset_2[visual_field][some_id]['features'].shape))
print(list(dataset_2[text_field][some_id]['features'].shape))
print(list(dataset_2[acoustic_field][some_id]['features'].shape))
print("Different modalities have different number of time steps!")

print(dataset_2[visual_field][some_id]['features'][0])
print(dataset_2[text_field][some_id]['features'][0])
print(dataset_2[acoustic_field][some_id]['features'][0])
print("Visual and audio modalities are features while text is just words!")

# use torchtext to get word embeddings


# functions of a dataset
# 1. add another sequences, the value of the recipe may be URL (if seqs not exist) or path (if seqs exist)
label_field = 'CMU_MOSI_Opinion_Labels'
label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
dataset_2.add_computational_sequences(label_recipe, destination=None)

# 2. align the other seqs (include label_field) to label_field
dataset_2.align(label_field)


# check train/test/dev
storage_files = ['train', 'dev', 'test']
for f in storage_files:
    pth = os.path.join(DATA_PATH, f)
    with open(pth, 'rb') as d:
       data = pickle.load(d)
       print(len(data))
