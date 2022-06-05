import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
'''mfcc = np.load('mfcc_tagtraum_clean.npy')
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

edges = np.ones((nbr_graph, nbr_node, nbr_node))#random.randint(0, 2, (nbr_graph, nbr_node, nbr_node))
e = np.zeros([30, 100, 100])
res = []
for k in tqdm(range (30)) :
    for i in range(100) :
        for j in range(100):
            a = feat[:,k,i]
            b = feat[:, k, j]
            e[k, i, j] = cosine(a,b)
            res.append(e[k, i, j])

            if abs(e[k, i, j]) < 0 :
                e[k, i, j]= 0




plt.hist(res, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of edges repartition")
'''
label = np.load('graph_label.npy')
edge = np.load('graph_edges.npy')

mean_s = []
mean_e = []
std_e_lst = []
std_s_lst = []

plt.figure()
plt.hist(np.reshape(label,(-1)), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of classes repartition")
plt.show()
'''w = np.histogram(np.reshape(label, (-1)),15)'''
w = []
for c in tqdm(range(15)):
    lab_idx = np.where(label == c)
    w.append(len(np.reshape(lab_idx[0], (-1)))/3000)


for c in tqdm(range(13)):
    similarity = []
    lab_idx = np.where(label == c)
    #w.append(len(np.reshape(lab_idx), (-1)))
    for i in lab_idx[2]:
        for j in lab_idx[2]:
            similarity.append(edge[lab_idx[1], i, j])

    s = np.reshape(np.array(similarity), (-1))
    e = np.reshape(edge, (-1))
    ids = np.where(s != 1)
    ide = np.where(e != 1)
    m_s = np.mean(s[ids])
    m_e = np.mean(e[ide])
    std_s = np.std(s[ids])
    std_e = np.std(e[ide])
    std_s_lst.append(std_s)
    std_e_lst.append(std_e)
    mean_s.append(m_s)
    mean_e.append(m_e)

plt.figure()
plt.plot(mean_e-std_e, color = 'red')
plt.plot(mean_e, color = 'red')
plt.plot(mean_e+std_e, color = 'red')
plt.plot(mean_s, color = 'blue')
plt.plot(mean_s+std_s,color = 'blue')
plt.plot(mean_s-std_s, color = 'blue')
plt.show()


'''plt.hist(similarity, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of edges repartition")'''
#plt.show()



