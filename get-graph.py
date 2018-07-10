import sys
from scipy.sparse import coo_matrix
from scipy.io import mmread
from scipy import sparse
import tables
import time
import numpy as np
import pickle

def load_sparse_mat(name, filename='store.h5'):
    with tables.open_file(filename) as f:
        # get nodes
        attributes = []
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            attributes.append(getattr(f.root, f'{name}_{attribute}').read())

    # construct sparse matrix
    M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3])
    return M



def readDicFromFile(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def getSubMatrix(co_occur_matrix, vocabulary, sub_vocabulary):
    idxs = list(map(lambda x: vocabulary.get(x), sub_vocabulary))
    idxs = list(filter(lambda x: x is not None, idxs))
    return co_occur_matrix[idxs, :].tocsc()[:, idxs].todense()



if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python get-graph.py <vocabulary_in> <matrix_in> [<list of words>]')
        sys.exit(1)

    in_vocab = sys.argv[1]
    in_matrix = sys.argv[2]
    sub_vocab = sys.argv[3:]

    print(sub_vocab)

    matrix = load_sparse_mat("cooccur", in_matrix).tocsr()
    vocab = readDicFromFile(in_vocab)
    sb = getSubMatrix(matrix, vocab, sub_vocab)
    print(sb)

    

