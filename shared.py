import numpy as np
import pickle
import re
import json
import string
import tables
import math
import operator
import networkx as nx
import nltk

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from scipy import sparse
from statistics import median
from functools import reduce
from scipy.spatial.distance import cosine


def readDicFromFile(file_name):
    '''
        read dictionary from file
    '''
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def writeDicToFile(data, file_name):
    """Write vocabular dic to file
    
    Arguments:
        data {dict} -- [description]
        file_name {string} -- [description]
    """

    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def readFrequencyFile(file_name):
    d = {}
    with open(file_name) as f:
        for line in f:
            (key, val) = line.split()
            d[key] = val
    return d

def loadSparseM(filename, name = 'cooccur'):
    with tables.open_file(filename) as f:
        # get nodes
        attributes = []
        for attribute in ('data', 'indices', 'indptr', 'shape'):
            attributes.append(getattr(f.root, f'{name}_{attribute}').read())

    # construct sparse matrix
    M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3])
    return M

def storeSparseMat(M, filename, name = 'cooccur'):
    """Store CSR sparse matrix to file with HDF5 format
    
    Arguments:
        M {csr_matrix} -- [description]
        filename {string} -- [description]
    
    Keyword Arguments:
        name {str} -- [description] (default: {'cooccur'})
    """

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


def normalize_text(s, lower_case = True):
    """Remove punctuation, spaces, articles
    
    Arguments:
        s {string} -- [description]
    
    Returns:
        [string] -- [description]
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower() if lower_case else text
        #return text.lower()
    
    def remove_stop_words(text):
        return re.sub(r'\b(as|to|that|and|of)\b', ' ', text)

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def myTokenizerGensimCompatible(text, token_min_len, token_max_len, lower):
    return myTokenizer(text)


def myTokenizer(text, POS=False, ACCUM_POS=True):
    """[summary]
    
    Arguments:
        text {[type]} -- [description]
    
    Keyword Arguments:
        POS {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- list of tokens
    """
    stop_words = "a an the".split()
    exclude_punctuation = set(string.punctuation)
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    tagged_filtered = filter(lambda v: v[0] != v[1] and v[0] not in exclude_punctuation and v[0].lower() not in stop_words, tagged)
    # replace cardinals CD with <CD>
    # replace proper nouns (NNP, NNPS) coming one after the other as single <NNP>
    def mapF(v):
        if (v[1] == 'NNP'): return '<NNP>' if not POS else (v[0], '<NNP>')
        if (v[1] == 'NNPS'): return '<NNP>' if not POS else (v[0], '<NNP>')
        if (v[1] == 'CD'): return '<NUM>' if not POS else (v[0], '<NUM>')
        return v[0].lower() if not POS else (v[0].lower(), v[1])    
    
    def reduceF(ss, v):
        if (not ACCUM_POS): return ss + [v]
        if (len(ss) == 0): return ss + [v]
        if (ss[-1][1] == v[1] and v[1] == 'NNP'): 
            ss[-1] = (ss[-1][0] + ' ' + v[0], ss[-1][1])
            return ss
        return ss + [v]

    return list(map(mapF, reduce(reduceF, tagged_filtered, [])))
    #return reduce(reduceF, map(mapF, tagged_filtered), [])


def getSubMatrix(co_occur_matrix, idxs):
    '''Extract square submatrix by list of indicies
    @co_occur_matrix  sparse co-occur matrix CSR
    @idxs  list of indicies to keep
    @return CSR matrix
    '''
    #idxs = list(map(lambda x: vocabulary.get(x), sub_vocabulary))
    #idxs = list(filter(lambda x: x is not None, idxs))
    return co_occur_matrix[idxs, :].tocsc()[:, idxs].tocsr()


def squadIterator(data_file, mode = 'train'):
    """Creates Iterator over SQuAD dataset. Yields tuples (id, context, question, answer)
    
    Arguments:
        data_file {string} -- file name
        mode {string} -- either 'train' or 'dev'
    """

    for v in flatten_json(data_file, mode):
        yield v


def flatten_json(data_file, mode):
    """Flatten each article in training data."""
    with open(data_file) as f:
        data = json.load(f)['data']
    rows = []
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                if mode == 'train':
                    answer = answers[0]['text']  # in training data there's only one answer
                    answer_start = answers[0]['answer_start']
                    answer_end = answer_start + len(answer)
                    #rows.append((id_, context, question, answer, answer_start, answer_end))
                    rows.append((id_, context, question, answer))
                else:  # mode == 'dev'
                    answers = [a['text'] for a in answers]
                    rows.append((id_, context, question, answers))
    return rows

def squadIteratorGraph(data_file, squad_vocab, matrix, freq_dict):
    """Creates iterator (Dense matrix, Map<word, (idx, QuestionFlag, AnswerFlag)>, Answer, Question, Paragraph)
    
    Arguments:
        data_file {string} -- dataset file path
        squad_vocab {Map<word, idx>} -- SQuAD dictionary
        matrix {csr_matrix} -- full co-occurence matrix
        freq_dict {dict} -- word corpus frequence
    """

    iter = squadIterator(data_file)
    for (id, paragraph, question, answer) in iter:
        
        yield processOneExample(squad_vocab, matrix, answer, question, paragraph, freq_dict)
        #yield (graph_matrix, idxs), answer, question, paragraph


def processOneExample(squad_vocab, matrix, answer, question, paragraph, freq_dict):
    """Process on example from SQuaD dataset. returns (Dense matrix, Map<word, (idx, QuestionFlag, AnswerFlag)>, Answer, Question, Paragraph)
    
    Arguments:
        answer {[type]} -- [description]
        question {[type]} -- [description]
        paragraph {[type]} -- [description]
        freq_dict {dict} -- word corpus frequence
    
    Returns:
        [type] -- [description]
    """

    idxs = {}
    answer_words = normalize_text(answer).split()
    for w in normalize_text(paragraph).split():
        if w not in squad_vocab: continue
        idxs[w] = {
            "idx": idxs.get(w)["idx"] if (idxs.get(w) is not None) else len(idxs),
            "word": w,
            "vocab_idx": squad_vocab[w], 
            "in_question": False,
            "in_answer": w in answer_words,
            "freq": freq_dict[w] if (freq_dict.get(w) is not None) else 0
        }
        
        #idxs[w] = (squad_vocab[w], False)
    for w in normalize_text(question).split():
        if w not in squad_vocab: continue
        #idxs[w] = (squad_vocab[w], True)
        idxs[w] = {
            "idx": idxs.get(w)["idx"] if (idxs.get(w) is not None) else len(idxs),
            "word": w,
            "vocab_idx": squad_vocab[w], 
            "in_question": True,
            "in_answer": w in answer_words,
            "freq": freq_dict[w] if (freq_dict.get(w) is not None) else 0
        }

    idxs = sorted(idxs.values(), key=lambda x: x["idx"])

    only_idxs = [v["vocab_idx"] for v in idxs]
    graph_matrix = getSubMatrix(matrix, only_idxs).todense()
    assert(len(idxs) == graph_matrix.shape[0])
    
    return (graph_matrix, idxs), answer, question, paragraph


def mkGraph(adjacency_matrix):

    # make distance matrix
    max = adjacency_matrix.max() + 1
    rows, cols = np.where(adjacency_matrix >= 1)
    edges = list(zip(rows.tolist(), cols.tolist()))
    nodes = set([n1 for n1, n2 in edges] + [n2 for n1, n2 in edges])

    gr = nx.Graph()
    for node in nodes:
        gr.add_node(node)
    for a,b in edges:
        gr.add_edge(a, b, weight=max - adjacency_matrix[a,b])
    return gr


def nearestNodes(graph, source_node_idxs, target_node_idxs, avg = True):
    """Measure distance from group of node to other nodes
    
    Arguments:
        graph {[type]} -- [description]
        source_node_idxs {[type]} -- [description]
        target_node_idxs {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    res = {}
    for node_target in target_node_idxs:
        accum = []
        for node_source in source_node_idxs:
            accum.append(nx.shortest_path_length(graph, source=node_source, target=node_target, weight="weight"))
        if avg: 
            res[node_target] = sum(accum) / float(len(accum))
        else:
            res[node_target] = median(accum)
    #idxs = sorted(idxs.values(), key=lambda x: x["idx"])
    return sorted(res.items(), key=operator.itemgetter(1))

def drawGraphWithLabels(adjacency_matrix, mylabels):
    """Draw graph
    
    Arguments:
        adjacency_matrix {Matrix} -- co-occur matrix
        mylabels {Map<idx, word>} -- labels distionary
    """

    gr = mkGraph(adjacency_matrix)

    # leave only used labels
    labels = { your_key: mylabels[your_key] for your_key in nodes }
    nx.draw_random(gr, node_size=2000, labels=labels, with_labels=True)
    plt.show()


def initEdgeVectors(adjacency_matrix, q_idxs):
    X = np.zeros(shape=(0, 4))
    for row in adjacency_matrix[:, q_idxs]:
        vector = np.array([.0, .0, .0, .0])
        for v in np.asarray(row).reshape(-1):
            vector += np.array([v, math.log(v) if v >= 1 else 0, math.pow(v, 2), math.sqrt(v) if v > 0 else 0 ])

        X = np.vstack((X, vector))
    
    assert(adjacency_matrix.shape[0] == X.shape[0])
    
    return X



def getDistanceMatrix(matrix, paragraph, question, vocab):
    """Calculate distance matrix and idx map
    
    Arguments:
        matrix {[type]} -- co-occur matrix
        paragraph {string} -- paragraph
        question {string} -- question
        vocab {Map Word Idx} -- [description]
    
    Returns:
        [type] -- (matrix, Map Word Idx)
    """

    rows = np.empty([0, matrix.shape[1]])
    dm_word_idx_map = {} # Map Word row_idx_in_distance_matrix
    # run over paragraph+question and check each word in VOCAB
    paragraph_words = myTokenizer(paragraph)
    question_words = myTokenizer(question)
    for w in paragraph_words + question_words:
        if w not in vocab: continue
        if w in dm_word_idx_map: continue
        # if word there then copy and append matrix's row with index VOCAB[word] to distanceMatrix
        v = matrix.getrow(vocab[w])
        #v_norm = v / v.multiply(v).sum()
        #v_norm = v / v.sum()
        rows = np.append(rows, v.todense().A, axis = 0)
        # and add  dm_word_idx_map[word] = idx of just inserted row 
        dm_word_idx_map[w] = len(rows) - 1
    

    # create matrix based on list of rows and multiply it on its transposed. we get cosine similarity
    
    dm = []
    for i in rows:
        r = []
        for j in rows:
            #r.append(euclideanDist(i, j))
            r.append(cosDist(i, j))

        dm.append(r)
    
    m = np.matrix(dm)
    return m, dm_word_idx_map
    



def spanDistance(span, question, distanceMatrix, dm_word_idx_map):
    """Distance between span and  question
    
    Arguments:
        span {[Word]} -- [description]
        question {[Word]} -- [description]
        distanceMatrix {[type]} -- [description]
        dm_word_idx_map {[type]} -- [description]
    
    Returns:
        number -- [description]
    """
    question_words = myTokenizer(question)
    span_words = myTokenizer(span)
    dss = []
    for w in span_words:
        if w not in dm_word_idx_map: 
            #print("skip unknown word:", w)
            continue # if span contains word out of vocabulary then skip it
        ds = _distancesOneToMany(w, question_words, distanceMatrix, dm_word_idx_map)
        dss.append(min(ds))
    
    return sum(dss) / float(len(dss))


def _distancesOneToMany(word, words, distanceMatrix, dm_word_idx_map):
    """List of distances from one word to set of words
    
    Arguments:
        word {Token} -- [description]
        words {[Token]} -- [description]
        distanceMatrix {[type]} -- [description]
        dm_word_idx_map {[type]} -- [description]
    
    Returns:
        [number] -- distances
    """

    ds = []
    for w in words:
        if w not in dm_word_idx_map:
            #print("skip unknown word:", w)
            continue
        i = dm_word_idx_map[word]
        j = dm_word_idx_map[w]
        ds.append(distanceMatrix[i,j])
    return ds

def cosDist(v1, v2):
    """Cosin distance between two verctors
    
    Arguments:
        v1 {[type]} -- vector
        v2 {[type]} -- vector
    
    Returns:
        [type] -- cos distance between vectors
    """

    # v1 = v1 / v1.sum()
    # v2 = v2 / v2.sum()
    # return v1.multiply(v2).sum()
    return cosine(v1, v2)

def euclideanDist(v1, v2):
    return np.linalg.norm(v1 - v2)


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def getSublists(lst, l):
    """ Get all sublist of difined length
    """
    return [lst[i:i+l] for i in range(len(lst) - l + 1)]

def sublist(lst1, lst2):
    """ Check whether lst1 is sublist of lst2
    """
    l = len(lst1)
    sublists = getSublists(lst2, l)
    return lst1 in sublists