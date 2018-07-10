import sys
import tables
import time
import numpy as np
from scipy import sparse
import gc


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
    M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3], dtype=np.uint16)
    return M


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python accum_dumps.py <out_accum_file> dump_file-1 [...dump-file-j]')
        sys.exit(1)

    out_matrix = sys.argv[1]
    dump_files = sys.argv[2:]

    start_time = time.time()

    # accum dumps
    matrix = load_sparse_mat("cooccur", dump_files[0])
    for dump_file in dump_files[1:]:
        #m = load_sparse_mat("cooccur", dump_file)
        matrix += load_sparse_mat("cooccur", dump_file)
        #del m
        gc.collect()
        print('current matrix size Mb: ', (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)  / (1024 * 1024))
        print('number elmenent ', matrix.count_nonzero())

    # store in one file
    store_sparse_mat(matrix, "cooccur", out_matrix)

    print("--- %s seconds ---" % (time.time() - start_time))

