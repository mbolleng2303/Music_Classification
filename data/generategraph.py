import numpy as np

mfcc = np.load('mfcc_tagtraum_clean.npy')
track_ids_mfcc = np.load('track_ids_tagtraum.npy')
chroma = np.load('chroma_tagtraum_clean.npy')
label = np.load('genre_tagtraum.npy')
nbr_graph = 30
nbr_node = 100
classes = np.unique(label)
nbr_classes = len(classes)
i = 0
label_int = np.zeros(len(label))
for lab in classes:
    idx = np.where(label == lab)
    label_int[idx] = int(i)
    i+=1
label = label_int
feat = np.reshape(mfcc, (3011, 300*12)).T
feat2 = np.reshape(chroma, (3011, 300*12)).T

feat = np.reshape(feat[0:3000], (-1, nbr_graph, nbr_node))
label = np.reshape(label[0:3000], (1, nbr_graph, nbr_node))
edges = np.ones((nbr_graph, nbr_node, nbr_node))#random.randint(0, 2, (nbr_graph, nbr_node, nbr_node))
np.save('graph_label', label)
np.save('graph_mfcc', feat)
np.save('graph_chroma', feat2)
np.save('graph_edges', edges)

