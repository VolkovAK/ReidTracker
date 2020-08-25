import numpy as np
import faiss
import pickle
import math
from pathlib import Path
import os
from scipy.optimize import linear_sum_assignment

class EmbeddingDB:
    def __init__(self, embedding_dim = 512, cache_mb=100000,
                database_dir='./data/faces', mode='flat',
                k_neighbors=15, thresh=1.5):


        self.embedding_dim = embedding_dim
        self.cache_mb = cache_mb
        self.database_dir = database_dir
        self.mode = mode
        self.k_neighbors = k_neighbors
        self.thresh = thresh

        self.index_path = None
        self.index = None

        if self.database_dir is not None:
            self.index_path = os.path.join(self.database_dir, "search.index")
            Path(database_dir).mkdir(parents=True, exist_ok=True)
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)

        if self.index is None:
            if self.mode == 'flat':
                self.index = faiss.index_factory(self.embedding_dim, "IDMap,Flat")
            elif self.mode == 'fast':
                self.nlist = 100
                self.quantizer = faiss.IndexFlatL2(self.embedding_dim)
                self.sub_index = faiss.IndexIVFFlat(self.quantizer, self.embedding_dim, self.nlist)
                self.index = faiss.IndexIDMap(sub_index)
                self.index.nprobe = 10
            elif self.mode == 'cosine':
                self.index = faiss.index_factory(self.embedding_dim, "IDMap,Flat,L2norm", faiss.METRIC_INNER_PRODUCT)

    def save(self):
        faiss.write_index(self.index, self.index_path)

    def train(self, embs):
        self.index.train(embs)
        assert index.is_trained


    def add(self, embs, ids, unique_ids=None):
        self.index.add_with_ids(embs, ids)

    def size(self):
        return self.index.ntotal

    def reset(self):
        self.index.reset()

    def remove(self, ids):
        self.index.remove_ids(ids)

    def find_knn(self, embs):
        return self.index.search(embs, self.k_neighbors)

    def find_ids(self, embs, thresh=1.5):
        D, I = self.find_knn(embs)

        if self.mode == 'cosine':
            I[D < (1 - self.thresh)] = 0
            D[D < (1 - self.thresh)] = 0
        else:
            I[D > self.thresh] = 0
            D[D > self.thresh] = self.thresh
            D = self.thresh - D


        classes = []
        confidences = []

        for i, d in zip(I, D):
            bcount = np.bincount(i, weights=d)
            class_id = bcount.argmax()
            class_count = np.bincount(i)
            confidence = bcount[class_id]/class_count[class_id] if class_count[class_id] != 0 else 0 

            classes.append(class_id)
            confidences.append(confidence)

        return classes, confidences


    def cost_matrix(self, embs):
        D, I = self.find_knn(embs)

        if self.mode == 'cosine':
            I[D < (1 - self.thresh)] = 0
            D[D < (1 - self.thresh)] = 0
        else:
            I[D > self.thresh] = 0
            D[D > self.thresh] = self.thresh
            D = self.thresh - D


        all_classes = np.unique(I)
        max_class = np.max(I)
        distance_matrix = np.zeros((embs.shape[0], all_classes.shape[0]))


        for row, (i, d) in enumerate(zip(I, D)):
            emb_confidences = np.zeros(max_class + 1)
            bcount = np.bincount(i, weights=d)
            class_count = np.bincount(i) + 0.00001
            confidence = bcount / class_count
            emb_confidences[:len(confidence)] = confidence
            emb_confidences = emb_confidences[all_classes]
            distance_matrix[row, :] = 1 - emb_confidences

        return distance_matrix, list(all_classes)



    def try_to_restore_track_id(self, embs):
        classes, confs = self.find_ids(embs)
        bcount = np.bincount(classes) # omit zeroth class since its empty class label
        class_id = bcount.argmax()
        if class_id > 0: # there is some class candidate
            likeliness = bcount[class_id] / embs.shape[0] # share of top candidate in total amount of features
            if likeliness > 0.8: # more than 80% of candidates have the same class id
                return class_id
        return None
