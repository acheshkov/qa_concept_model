
import torch
import math
import numpy as np
import datetime
from torch.autograd import Variable
from model_cnn_v2 import MyCNN2
from dataset import MyDataset
from functools import reduce
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value, log_histogram
from shared import (squadIteratorGraph, readDicFromFile, loadSparseM)
import argparse
import logging

def parseArgs():
    '''Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        description='Train model'
    )
    parser.add_argument('--global_vocab', required = True,
                        help='path to vocab file.')

    parser.add_argument('--matrix', required = True,
                        help='path to co-occurence matrix file')

    parser.add_argument('--squad_train', default="train-v1.1.json",
                        help='path to co-occurence matrix file')
    
    parser.add_argument('--squad_eval', default="dev-v1.1.json",
                        help='SQuAD evaluation dataset')

    parser.add_argument('--span_len', default = 20, type=int,
                        help='length of SPAN')

    parser.add_argument('--train_examples_len', default = 200, type=int,
                        help='how much examples use for training')

    parser.add_argument('--eval_examples_len', default = 150, type=int,
                        help='number of examples from SQUAD  use to eval')

    parser.add_argument('--batch_size', default = 13, type=int,
                        help='train batch size')

    parser.add_argument('--num_epochs', default = 1000, type=int,
                        help='epochs number')


    return parser.parse_args()
    

if __name__ == '__main__': 
    args = parseArgs()

    logging.basicConfig(
        format='%(asctime)s %(message)s', 
        level=logging.DEBUG,
        datefmt='%m/%d/%Y %I:%M:%S'
    )
    log = logging.getLogger(__name__)
    log.info(vars(args))

    
    log.info('read global vocabs from file...')
    vocab = readDicFromFile(args.global_vocab)
    log.info('read matrix from file...')
    matrix = loadSparseM(args.matrix)
    model = MyCNN2()
    SPAN_LEN = args.span_len
    learning_rate = 0.001
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    train_examples_len = args.train_examples_len
    eval_examples_len = args.eval_examples_len

    # Dataset for evaluation
    log.info(f'Dataset for evaluation, {eval_examples_len} examples...')
    dsEval = MyDataset(args.squad_eval, eval_examples_len, vocab, matrix, span_len=SPAN_LEN)
    dataloaderEval = DataLoader(dsEval, batch_size=1, shuffle=True, num_workers=0)

    # Dataset for train
    log.info(f'Dataset for train, {train_examples_len} examples...')
    ds = MyDataset(args.squad_train, train_examples_len, vocab, matrix, span_len=SPAN_LEN)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # tensorboard logger config https://github.com/TeamHG-Memex/tensorboard_logger
    log.info('Configure tensorboard logger...')
    dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    configure(f"runs/{dt}", flush_secs=5)
    
    
    [pos_count, neg_count] = reduce(lambda accum, v: (accum[0] + 1, accum[1]) if v[1].tolist()[0] == 1 else  (accum[0], accum[1] + 1), ds, [0,0])
    pos_weight = pos_count/len(ds)
    neg_weight = neg_count/len(ds)
    log.info(f'TRUE example weight: {pos_weight}. FALSE - {neg_weight}')

    log.info(f'Start training, {num_epochs} epoch; {len(ds)} samples...')
    for epoch in range(num_epochs):
        log.info(f'Epoch: {epoch}')
        model.trainModel(
            dataloader, 
            pos_count/len(ds), 
            neg_count/len(ds), 
            log_value
        )
        if (epoch + 1) % 10 == 0: 
            log_value('accuracy', model.evaluateModel(dataloaderEval), epoch)


    
