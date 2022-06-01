import numpy as np

mfcc = np.load('mfcc_tagtraum_clean.npy')
track_ids_mfcc = np.load('track_ids_tagtraum.npy')
chroma = np.load('chroma_tagtraum_clean.npy')
label = np.load('genre_tagtraum.npy')



classes = np.unique(label)
i = 0
label_int = np.zeros(len(label))
for lab in classes:
    idx = np.where(label == lab)
    label_int[idx] = int(i)
    i+=1

np.save('label', label_int)
a = 2



