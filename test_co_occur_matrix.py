import sys
import math

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


def test(w1, w2, iterator):
    d1 = {}
    d2 = {}
    all_keys = {}
    sum_w1 = 0
    sum_w2 = 0

    for (word, context) in iterator:
        if word not in [w1, w2]: continue
        for cword in context:
            if (word == w1):
                if cword in d1: 
                    d1[cword] = d1[cword] + 1 
                else:
                    d1[cword] = 1
                sum_w1+=1

            if (word == w2):
                if cword in d2: 
                    d2[cword] = d2[cword] + 1
                else:
                    d2[cword] = 1
                sum_w2+=1
                
            all_keys[cword] = 1
    
    s = 0
    #print(d1)
    #print(d2)
    print(d1[w2])
    for key in all_keys:
        #print(key)
        a = d1[key]/sum_w1 if key in d1 else 0
        b = d2[key]/sum_w2 if key in d2 else 0
        s += math.pow(a - b, 2)

    return math.sqrt(s)

        
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python test-co-occur-matrix.py w1 w2 context_size <text_file_in>')
        sys.exit(1)

    w1 = sys.argv[1]
    w2 = sys.argv[2]
    context_size = int(sys.argv[3])
    in_f = sys.argv[4]
    
    iter = iterCorpus(in_f, context_size)
    #cooccurrence_matrix = mkCoOccurMatrix(iter, vocab)
    distance = test(w1, w2, iter)
    print("Distance: ", distance)