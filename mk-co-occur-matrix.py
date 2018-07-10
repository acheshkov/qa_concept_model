import sys
import gc
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from scipy import sparse
import tables
import time
import numpy as np
import pickle


def store_sparse_mat(M, name, filename='store.h5'):
    print(M.__class__)
    assert(M.__class__ == sparse.csr.csr_matrix), 'M must be a csr matrix'
    with tables.open_file(filename, 'a') as f:
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            full_name = f'{name}_{attribute}'

            # remove existing nodes
            try:  
                n = getattr(f.root, full_name)
                n._f_remove()
            except AttributeError:
                pass

            # add nodes
            arr = np.array(getattr(M, attribute))
            atom = tables.Atom.from_dtype(arr.dtype)
            ds = f.create_carray(f.root, full_name, atom, arr.shape)
            ds[:] = arr

def load_sparse_mat(name, filename='store.h5'):
   
    with tables.open_file(filename) as f:

        # get nodes
        attributes = []
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            attributes.append(getattr(f.root, f'{name}_{attribute}').read())

    # construct sparse matrix
    M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3])
    return M

def sumCSRMatrix(m1, m2):
    mdim1, mdim2 = m1.get_shape()
    tdim1, tdim2 = m2.get_shape()
    #print(mdim1, mdim2)
    #print(tdim1, tdim2)
    tmp1 = csr_matrix((m1.data, m1.indices, m1.indptr), shape = (max(mdim1, tdim1), max(mdim2, tdim2)))
    tmp2 = csr_matrix((m2.data, m2.indices, m2.indptr), shape = (max(mdim1, tdim1), max(mdim2, tdim2)))
    return tmp1 + tmp2



# dump version
# callculates matrix and dump it to disk if threshold
def mkCoOccurMatrixDump(iterator, vocabulary, threshold_mb, file_name_prefix):
    DIM = len(vocabulary)
    #vocabulary = {}
    data = []
    row = []
    col = []
    dump_counter = 0
    matrix = coo_matrix(([], ([],[])), shape=(DIM,DIM), dtype=np.int32).tocsr()

    for (word, context) in iterator:
        if (word not in vocabulary): continue
        i = vocabulary[word]
        for cword in context:
            if (cword not in vocabulary): continue
            j = vocabulary[cword]
            data.append(1)
            row.append(i)
            col.append(j)
            
        if (sys.getsizeof(data) / (1024 * 1024) > 2048):
            print('matrix compression')
            
            tmp = coo_matrix((data, (row, col)), shape=(DIM,DIM))
            tmp.setdiag(0)
            tmp = tmp.tocsr()
            matrix = sumCSRMatrix(matrix, tmp)
            print('current matrix size Mb: ', (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)  / (1024 * 1024))
            gc.collect()
            
            data = []
            row = []
            col = []
            gc.collect()

        if ((matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / (1024 * 1024) > threshold_mb):
            # dump matrix to disk
            print('dump start')
            print('current matrix size Mb: ', (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)  / (1024 * 1024))
            store_sparse_mat(matrix, "cooccur", f'{file_name_prefix}_{dump_counter}')
            matrix = coo_matrix(([], ([],[])), shape=(DIM,DIM), dtype='i').tocsr()
            dump_counter+=1


    #print("Data array length:", len(data))
    #print("Data array in memory:", sys.getsizeof(data) / (1024 * 1024))
    #cooccurrence_matrix = coo_matrix((data, (row, col)))
    tmp = coo_matrix((data, (row, col)), shape=(DIM,DIM))
    tmp.setdiag(0)
    tmp = tmp.tocsr()
    cooccurrence_matrix = sumCSRMatrix(matrix, tmp)
    store_sparse_mat(cooccurrence_matrix, "cooccur", f'{file_name_prefix}_{dump_counter}')
    return None


# iterator, returns (line)
def _iterFile(file_name):
    for line in open(file_name):
        yield line

# iterator, returns (word, left context, right context)
def _iterWordContext(words_list, window_size = 2):
    l = len(words_list)
    for i, c in enumerate(words_list):
        mn = max(0, i - window_size)
        mx = min(l, i + window_size + 1)
        mn_i = max(0, i)
        mx_i = min(l, i + 1)
        yield (c, words_list[mn:mn_i], words_list[mx_i:mx])

def iterCorpus(file_name, window_size):
    for line in _iterFile(file_name):
        for (w, lc, rc) in _iterWordContext(line.split(), window_size):
            yield (w, lc + rc)

def writeDicToFile(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

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
        print('Usage: python mk-co-occur-matrix.py context_size <text_file_in> <vocabulary_in> <matrix_out> [<dump_threshold>]')
        sys.exit(1)

    context_size = int(sys.argv[1])
    in_f = sys.argv[2]
    out_matrix = sys.argv[4]
    in_vocabulary = sys.argv[3]
    dump_threshold = int(sys.argv[5] if len(sys.argv) > 5 else 3048)
    

    start_time = time.time()

    vocab = readDicFromFile(in_vocabulary)
    print('Vocabulary Length: ', len(vocab))
    iter = iterCorpus(in_f, context_size)
    mkCoOccurMatrixDump(iter, vocab, dump_threshold, out_matrix)
    print("--- %s seconds ---" % (time.time() - start_time))

