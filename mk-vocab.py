import sys
from scipy import sparse
import time
import numpy as np
import pickle

def mkVocab(iterator):
    
    vocabulary = {}
    
    for (word, context) in iterator:
        
        if (word not in vocabulary):
            i = vocabulary.setdefault(word, (len(vocabulary), 1))
        else:
            idx, freq = vocabulary[word]
            vocabulary[word] = (idx, freq + 1)
     
    return vocabulary


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


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python mk-vocab.py <text_file_in> <vocabulary_out> [top]')
        sys.exit(1)


    in_f = sys.argv[1]
    out_vocabulary = sys.argv[2]
    top = int(sys.argv[3] if len(sys.argv) > 3 else 1000000)
    print("top", top)
    
    start_time = time.time()
    iter = iterCorpus(in_f, 1)
    vocabulary = mkVocab(iter)
    
    
    print('Vocabulary len:', len(vocabulary))

    vocab_sorted = sorted(vocabulary.items(), key=lambda v: v[1][1], reverse=True)
    vocab_sorted = vocab_sorted[0: top]
    
    map_vocab = {word:i for i, (word, (idx, freq)) in enumerate(vocab_sorted)}
    #print(map_vocab)

    writeDicToFile(map_vocab, out_vocabulary)

    with open(f'freq_{out_vocabulary}', 'w') as thefile:
        for (word, (idx, freq)) in vocab_sorted:
            thefile.write(f'{word} {freq}\n')

    #print(cooccurrence_matrix.count_nonzero())
    print("--- %s seconds ---" % (time.time() - start_time))
    

