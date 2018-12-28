import cPickle 
import numpy as np


def read_node_embeddings():
    with open('mesh.embeddings', 'r') as f:
        lines = f.readlines()
    
    lines = lines[1:]
    node_to_vec = {}
    for line in lines:
        line = line.strip().split(" ")
        node_to_vec[int(line[0])] = line[1:]

    node_emb = []
    for i in xrange(len(node_to_vec.keys())):
        node_emb.append(node_to_vec[i])

    return np.array(node_emb, dtype=float)


def read_word_embeddings(word2index):
    ftypes = open('./wordembeddings/types.txt', 'r')
    words = ftypes.readlines()
    word_2_vectors = {}
    with open('./wordembeddings/vectors.txt','r') as f:
        i = 0
        for line in f:
            word_2_vectors[words[i].strip().lower()] = map(float, line.strip().split(" "))
            # print len(word_2_vectors[words[i].strip().lower()])
            i += 1

    random_emb = np.random.standard_normal(200).tolist()
    word_embeddings = [[] for i in range(len(word2index.keys()))]
    for word in word2index.keys():
        if word in word_2_vectors:
            word_embeddings[int(word2index[word])] += word_2_vectors[word] 
        else:
            word_embeddings[int(word2index[word])] += random_emb

    return np.array(word_embeddings)


