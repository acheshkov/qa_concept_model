import json
import sys
import numpy as np
import argparse
#from squad_iterator import squadIterator 
from shared import readDicFromFile, getSubMatrix, loadSparseM, storeSparseMat
import logging



def parseArgs():
    '''Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        description='Reduce size of co-occur matrix. Remove noise and use only SQuAD vocabulary'
    )
    parser.add_argument('--global_vocab', required = True,
                        help='path to vocab file.')

    # parser.add_argument('--squad_vocab', required = True,
    #                     help='path to squad dataset file.')

    parser.add_argument('--matrix', required = True,
                        help='path to co-occurence matrix file')

    parser.add_argument('--threshold', required = False, default = 0, type=int,
                        help='also remove elements <= threshold')

    parser.add_argument('--out', required = True,
                        help='output file reduced matrix')

    return parser.parse_args()
    

if __name__ == '__main__': 
    args = parseArgs()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    log.info(vars(args))

    log.info('read global vocabs from file...')
    glovalVocab = readDicFromFile(args.global_vocab)
    # log.info('read squad vocabs from file...')
    # squadVocab = readDicFromFile(args.squad_vocab)
    log.info('read matrix from file...')
    matrix = loadSparseM(args.matrix)

    # idxs_to_keep = []
    # for w in glovalVocab:
    #     if w in squadVocab:
    #         idxs_to_keep.append(glovalVocab[w])
    
    # log.info('reduce matrix, we want keep %s indices...', len(idxs_to_keep))
    # matrix_new = getSubMatrix(matrix, idxs_to_keep)
    # log.info('Reduced matrix shape is %s', matrix_new.shape)
    


    if (args.threshold > 0):
        log.info('Threshold enabled and is %s', args.threshold)
        log.info('Non zero element before noise reduction: %s', matrix.count_nonzero())
        matrix.data *= matrix.data > args.threshold
        matrix.eliminate_zeros()
        log.info('Non zero element after noise reduction: %s', matrix.count_nonzero())


    log.info('save matrix to disk...')
    storeSparseMat(matrix, args.out)

    
