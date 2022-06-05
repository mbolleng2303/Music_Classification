import numpy as np
from tqdm import tqdm
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
def cosine(input1, input2):
    res = np.dot(input1, input2)/(np.linalg.norm(input1)*np.linalg.norm(input2))
    return res
np.save('graph_label', label)
np.save('graph_mfcc', feat)
np.save('graph_chroma', feat2)


edges = np.ones((nbr_graph, nbr_node, nbr_node))#random.randint(0, 2, (nbr_graph, nbr_node, nbr_node))
e = np.zeros([30, 100, 100])

res = []
for k in tqdm(range (30)) :
    for i in range(100) :
        for j in range(100):
            a = feat[:,k,i]
            b = feat[:, k, j]
            #e[k, i,j] = cosine(a,b)
            res.append(e[k, i, j])

            '''if e[k, i, j] < 0.11:
                e[k, i, j] = 0'''
            a = label[0,k,i]
            b = label[0,k,j]
            if a == b :
                e[k,i,j]=1






import matplotlib.pyplot as plt

'''#e = np.reshape(e,[1,-1])
plt.plot([0.11,0.11], [0, 4500])
plt.legend('treshold')'''
plt.hist(res, bins='auto')  # arguments are passed to np.histogram

plt.title("Histogram of edges repartition")
plt.xlim([0,0.3])
plt.xlabel('cosine similarity')
plt.ylabel('Number')
plt.show()
np.save('graph_edges', e)

""""----------------------------------------------------------------------------------------------------------------------------------------"""



import numpy as np

mfcc = np.load('mfcc_tagtraum_clean.npy')
track_ids_mfcc = np.load('track_ids_tagtraum.npy')
chroma = np.load('chroma_tagtraum_clean.npy')
label = np.load('genre_tagtraum.npy')
nbr_graph = 3011
nbr_node = 12
classes = np.unique(label)
nbr_classes = len(classes)
i = 0
label_int = np.zeros(len(label))
for lab in classes:
    idx = np.where(label == lab)
    label_int[idx] = int(i)
    i+=1
label = label_int
feat = np.reshape(mfcc, (3011, 300, 12)).T
feat2 = np.reshape(chroma, (3011, 300, 12)).T
# feat = np.reshape(feat[0:3000], (-1, nbr_graph, nbr_node))
label = np.reshape(label, (1, 3011))
edges = np.ones((nbr_graph, nbr_node, nbr_node))#random.randint(0, 2, (nbr_graph, nbr_node, nbr_node))
np.save('graph_label_a', label)
np.save('graph_mfcc_a', feat)
np.save('graph_chroma_a', feat2)
np.save('graph_edges_a', edges)

