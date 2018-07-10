import json
import sys
import numpy as np
import argparse
#from squad_iterator import squadIterator 
from shared import readDicFromFile, writeDicToFile, normalize_text, squadIterator
import logging



def parseArgs():
    '''
    Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        description='Make Squad vocabulary. Result is intersection of GlobalVocab and all unique words in dataset.'
    )
    parser.add_argument('--global_vocab', required = True,
                        help='path to vocab file.')

    parser.add_argument('--squad_dataset', required = True,
                        help='path to squad dataset file.')

    parser.add_argument('--out', required = True,
                        help='output file vocabulary')

    return parser.parse_args()
    

def squadAllWordsIterator(file_name): 
    '''
    Iterator over all words in SQuAD dataset
    '''
    squad_iter = squadIterator(file_name, 'train')
    for (id, p, q, a) in squad_iter:
        for w in normalize_text(p).split() + normalize_text(q).split() + normalize_text(a).split():
            yield w


def mkVocabulary(file_name):
    vocabulary = {}
    for word in squadAllWordsIterator(file_name):
        if (word not in vocabulary):
            i = vocabulary.setdefault(word, (len(vocabulary), 1))
        # else:
        #     idx, freq = vocabulary[word]
        #     vocabulary[word] = (idx, freq + 1)
    return vocabulary
            

if __name__ == '__main__': 
    args = parseArgs()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    log.info(vars(args))

    log.info('start build squad vocabulary...')
    squadVocab = mkVocabulary(args.squad_dataset)
    log.info('reading global vocabulary from file...')
    glovalVocab = readDicFromFile(args.global_vocab)

    log.info('Global vocabulary size: %s', len(glovalVocab))
    for w in list(glovalVocab):
        if (w not in squadVocab): del glovalVocab[w]

    log.info('reindex...')
    squad_vocab = {word:i for i, word in enumerate(glovalVocab)}
    
    log.info('SQuAD vocabulary size: %s', len(squad_vocab))
    log.info('save to file...')
    writeDicToFile(squad_vocab, args.out)
