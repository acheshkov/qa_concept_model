from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad 
from shared import storeSparseMat, loadSparseM, squadIterator, getDistanceMatrix, myTokenizer, getSublists, sublist, cosDist, _distancesOneToMany
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import numpy as np
import os.path
import torch
import time


class MyDataset(Dataset):
    """Squad Dataset"""

    def __init__(self, squad_file, num_examples, vocab, coocur_matrix, span_len = 15, file_name='distance_matrix'):
        """
        Args:
        """
        self.file_name = file_name
        self.dm = DistanceMatrix()
        self.dm.load(file_name)
        self.SPAN_LEN = span_len
        self.dataset = []

        start_time = time.time()
        for i, (id, paragraph, question, answer) in zip(range(num_examples), squadIterator(squad_file)):
            self.transform(coocur_matrix, vocab, paragraph, question, answer)
            #print(f"-----{i}-----")
        print("--- %s time ---" % (time.time() - start_time))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        l = len(self.dataset[idx][2])
        par_l = len(self.dataset[idx][3]) # paragraph len
        pad_size = 300 - par_l
        paragraph_tensor = pad(torch.FloatTensor(self.dataset[idx][3]).view(1, par_l), (0, pad_size)).data
        res = torch.FloatTensor(self.dataset[idx][2]).view(1, l), torch.ByteTensor([self.dataset[idx][0]]), paragraph_tensor
        return res

    def _getDistanceMatrix(self, matrix, paragraph, question, vocab):
        # if we have all distances then return it
        dm_word_idx_map = {}
        unique_words = []
        paragraph_words = myTokenizer(paragraph)
        question_words = myTokenizer(question)

        # filter tokens and 
        for w in paragraph_words + question_words:
            if w not in vocab: continue
            if w in dm_word_idx_map: continue
            dm_word_idx_map[w] = len(dm_word_idx_map) - 1
            unique_words.append(w)
        
        matrix_raw = []
        
        for i, w1 in enumerate(unique_words):
            row = []
            self.dm.startBulk() # start bulk insert mode
            for j, w2 in enumerate(unique_words):
                #if (j < i): continue # we can skip because matrix is symmetric
                cosDistance = self.dm.get(vocab[w1], vocab[w2]) # check if already calculated
                if (cosDistance is None or cosDistance == 0):
                    if (w1 == w2): continue # skip. distance is zero
                    
                    # ''' if distance not exist we will call method to get all distances
                    # and add this values to our distance matrix
                    # '''
                    # print("distance not exist we will call method to get all distances of sentense")
                    # self.calcAndAddDistances(matrix, paragraph, question, vocab)
                    # cosDistance = self.dm.get(vocab[w1], vocab[w2])
                    
                    
                    v1 = matrix.getrow(vocab[w1]).todense().A[0]
                    v2 = matrix.getrow(vocab[w2]).todense().A[0]
                    cosDistance = cosDist(v1.astype(np.int64), v2.astype(np.int64))
                    self.dm.addBulk(vocab[w1], vocab[w2], cosDistance)

                row.append(cosDistance)
                
            matrix_raw.append(row)
            self.dm.commitBulk() # finish bulk insert mode 
        
        
        distanceMatrix = np.matrix(matrix_raw)
        return distanceMatrix, dm_word_idx_map


    def calcAndAddDistances(self, matrix, paragraph, question, vocab):
        m, idxs = getDistanceMatrix(matrix, paragraph, question, vocab)
        print(f"calcAndAddDistances distanceMatrix {len(m)}")
        data = []
        for w1, idx1 in idxs.items():
            for w2, idx2 in idxs.items():
                oldVal = self.dm.get(vocab[w1], vocab[w2])
                if oldVal != None and oldVal > 0: continue
                data.append((vocab[w1], vocab[w2], m[idx1, idx2]))
        
        print(f"{len(data)} new values will be added")
        self.dm.addMany(data)
                

    def transform(self, matrix, vocab, paragraph, question, answer):
        """ :: (Map (Int, Int) -> Float) ->  Map Token Int -> [Token] -> [Token] -> [Token] -> ()
        """


        distanceMatrix, dm_word_idx_map = self._getDistanceMatrix(matrix, paragraph, question, vocab)
        p_tokens = myTokenizer(paragraph, POS=True, ACCUM_POS=False)
        q_tokens = myTokenizer(question, POS=True, ACCUM_POS=False)
        a_tokens = myTokenizer(answer, POS=True, ACCUM_POS=False)
        values = []
        tokens = []
        if (len(q_tokens)) > 300: return # we skip sentenses longer than 300 tokens
        for (w, pos) in p_tokens: 
    
            if pos == '<NNP>':
                if (w, pos) in q_tokens:
                    values.append(0)
                else:
                    without_nnp = list(map(lambda v: v[0] if v[1] != '<NUM>' else '<NUM>', filter(lambda v: v[1] not in ['<NNP>'], q_tokens)))
                    ds = _distancesOneToMany('<NNP>', without_nnp, distanceMatrix, dm_word_idx_map)
                    v = min(ds)
                    values.append(v)
                tokens.append(pos + ' ' + w)
            elif pos == '<NUM>':
                if (w, pos) in q_tokens:
                    values.append(0)
                else:
                    without_nums = list(map(lambda v: v[0] if v[1] != '<NNP>' else '<NNP>', filter(lambda v: v[1] not in ['<NUM>'], q_tokens)))
                    ds = _distancesOneToMany('<NUM>', without_nums, distanceMatrix, dm_word_idx_map)
                    v = min(ds)
                    values.append(v)
                tokens.append(pos  + ' ' + w)
            else:
                
                #if w in dm_word_idx_map: continue
                #print(w, pos)
                if w not in vocab: continue
                ts = list(map(lambda v: v[0], filter(lambda v: v[1] not in ['<NNP>', '<NUM>'], q_tokens)))
                ds = _distancesOneToMany(w, ts, distanceMatrix, dm_word_idx_map)
                v = min(ds)
                values.append(v)
                tokens.append(w)


        p_spans = getSublists(p_tokens, self.SPAN_LEN)
        v_spans = getSublists(values, self.SPAN_LEN)

        train_smpls = list(map(lambda v: [sublist(a_tokens, v[0]), v[0], v[1]], zip(p_spans, v_spans)))
        train_smpls = list(map(lambda v: v + [values], train_smpls))
        self.dataset = self.dataset + train_smpls


    
    def saveDistanceMatrixToDisc(self):
        self.dm.save(self.file_name)




class DistanceMatrix:
    """Distance Matrix  that can be stored on disk"""
    def __init__(self):
        self.m = None
    
    def save(self, file_name):
        if (self.m is None): return
        storeSparseMat(self.m, file_name, 'dmatrix')

    def load(self, file_name):
        if not os.path.isfile(file_name): return None
        self.m = loadSparseM(file_name, 'dmatrix') # returns CSR matrix
        
    
    def get(self, i, j):
        if (self.m is None): return None
        if (i > self.m.shape[0] or j > self.m.shape[0]): return None
        return self.m[i, j] or self.m[j, i] # matrix is symmetric 

    def startBulk(self):
        #print('start bulk transaction')
        self.buff_data = []
        self.buff_rows = []
        self.buff_cols = []

    def commitBulk(self):
        #print(f'{len(self.buff_data)} commited')
        if (len(self.buff_data)) == 0: return;
        if (self.m is None): 
            data = []
            rows = []
            cols = []
        else: 
            coo = self.m.tocoo()
            data = coo.data.tolist()
            rows = coo.row.tolist()
            cols = coo.col.tolist()
        
        
        self.m = coo_matrix((data + self.buff_data, (rows + self.buff_rows, cols + self.buff_cols))).tocsr()
        self.buff_data = []
        self.buff_rows = []
        self.buff_cols = []
    
    def addMany(self, ss):
        """Bulk add operation
        Arguments:
            data {[(i, j, value)]} -- list of (i,j,value)
        """
        if (self.m is None): 
            data = []
            rows = []
            cols = []
        else: 
            coo = self.m.tocoo()
            data = coo.data.tolist()
            rows = coo.row.tolist()
            cols = coo.col.tolist()

        for i, j, value in ss:
            rows.append(i)
            cols.append(j)
            data.append(value)
        
            if (i != j): # for symetry
                rows.append(j)
                cols.append(i)
                data.append(value)

        self.m = coo_matrix((data, (rows, cols))).tocsr()

    def add(self, i, j, value):
        #print("add", i, j, value)
        if (self.get(i, j) == value): 
            #print(f"value {value} for ({i}, {j}) already exists")
            return

        if (self.get(i, j) != value and self.get(i, j) != None): 
            print(f"matrix already has value {self.get(i, j)} at ({i}, {j}). it will be replaced by {value}")
            self.m[i, j] = value
            self.m[j, i] = value
            return

        if (self.m is None): 
            data = []
            rows = []
            cols = []
        else: 
            coo = self.m.tocoo()
            data = coo.data.tolist()
            rows = coo.row.tolist()
            cols = coo.col.tolist()

        rows.append(i)
        cols.append(j)
        data.append(value)
        
        if (i != j): # for symetry
            rows.append(j)
            cols.append(i)
            data.append(value)

        self.m = coo_matrix((data, (rows, cols))).tocsr()

    def addBulk(self, i, j, value):
        if (self.get(i, j) == value): 
            #print(f"value {value} for ({i}, {j}) already exists")
            return

        oldValue = self.get(i, j)
        if (oldValue!= value and oldValue != None and oldValue > 0): 
            print(f"matrix already has value {oldValue} at ({i}, {j}). it will be replaced by {value}")
            self.m[i, j] = value
            self.m[j, i] = value
            return


        self.buff_rows.append(i)
        self.buff_cols.append(j)
        self.buff_data.append(value)
        
        if (i != j): # for symetry
            self.buff_rows.append(j)
            self.buff_cols.append(i)
            self.buff_data.append(value)
