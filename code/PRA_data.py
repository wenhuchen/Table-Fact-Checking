from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import time
import json
import numpy as np
import torch
import logging
import re

logger = logging.getLogger(__name__)

def get_batch(option, data_dir, vocab, max_seq_length=50, cutoff=-1):
    examples = []
    prev_sys = None
    num = 0
    
    if option == 'train':
        f = open('{}/train.tsv'.format(data_dir))
        balanced = True
    elif option == 'val':
        f = open('{}/dev.tsv'.format(data_dir))
        balanced = False
    elif option == 'test':
        f = open('{}/test.tsv'.format(data_dir))
        balanced = False
    elif option == 'simple_test':
        f = open('{}/simple_test.tsv'.format(data_dir))
        balanced = False    
    elif option == 'complex_test':
        f = open('{}/complex_test.tsv'.format(data_dir))
        balanced = False
    elif option == 'small_test':
        f = open('{}/small_test.tsv'.format(data_dir))
        balanced = False 
    else:
        raise ValueError("Unknown Data Split")

    #logger.info("Loading total {} tables".format(len(ids)))

    #with open('{}/all_programs.json'.format(ready_dir)) as f:
    #    data = json.load(f)
    pos_buf = []
    neg_buf = []
    examples = []
    for lid, line in enumerate(f):
        if lid > cutoff and cutoff > 0:
            break
        
        entry = line.strip().split('\t')
        index = int(entry[1].split('-')[-1])
        true_label = int(entry[2])
        pred_label = int(entry[3])
        statement = entry[4]
        program = entry[5]
        label = int(entry[6])

        stat = [vocab.get(_, 1) for _  in statement.split(' ')]
        r = [vocab.get(_, 1) for _  in program.split(' ') if len(_) > 0]

        input_ids = stat
        program_ids = [vocab["<CLS>"]] + r
        
        if len(input_ids) > max_seq_length - 1 or len(program_ids) > max_seq_length - 1:
            continue
        else:
            input_ids = input_ids + [vocab["<PAD>"]] * (max_seq_length - len(input_ids))
            program_ids = program_ids + [vocab["<PAD>"]] * (max_seq_length - len(program_ids))

            if balanced:
                if label == 0:
                    neg_buf.append((input_ids, program_ids, index, true_label, pred_label, label))
                else:
                    pos_buf.append((input_ids, program_ids, index, true_label, pred_label, label))
                
                if len(pos_buf) > 0 and len(neg_buf) > 0:
                    pos = pos_buf.pop(0)
                    neg = neg_buf.pop(0)
                    examples.append((pos[0], pos[1], pos[2], pos[3], pos[4], 1))
                    examples.append((neg[0], neg[1], neg[2], pos[3], pos[4], 0))
            else:
                examples.append((input_ids, program_ids, index, true_label, pred_label, label))

    all_input_ids = torch.tensor([f[0] for f in examples], dtype=torch.long)
    all_prog_ids = torch.tensor([f[1] for f in examples], dtype=torch.long)
    all_index = torch.tensor([f[2] for f in examples], dtype=torch.int32)
    all_true = torch.tensor([f[3] for f in examples], dtype=torch.int32)
    all_pred = torch.tensor([f[4] for f in examples], dtype=torch.int32)
    labels = torch.tensor([f[5] for f in examples], dtype=torch.float32)
    #true_labels = torch.tensor([f[3] for f in examples], dtype=torch.int)
    #pred_labels = torch.tensor([f[4] for f in examples], dtype=torch.int)
    #all_template_ids = torch.tensor([f[9] for f in examples], dtype=torch.long)
    return all_input_ids, all_prog_ids, labels, all_index, all_true, all_pred