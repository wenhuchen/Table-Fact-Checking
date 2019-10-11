import json
from collections import Counter
import random
from PRA_data import get_batch
from Transformer import Encoder, Decoder
import argparse
import os
import sys
from itertools import chain
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import torch.nn as nn
import numpy


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_simple_test', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_complex_test', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_small_test', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--emb_dim', type=int, default=128, help="the embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.2, help="the embedding dimension")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume previous run")
    parser.add_argument('--batch_size', type=int, default=512, help="the embedding dimension")
    parser.add_argument('--data_dir', type=str, default='../preprocessed_data_program/', help="the embedding dimension")
    parser.add_argument('--max_seq_length', type=int, default=100, help="the embedding dimension")
    parser.add_argument('--layer_num', type=int, default=3, help="the embedding dimension")
    parser.add_argument('--voting', default=False, action="store_true", help="the embedding dimension")
    parser.add_argument('--id', default="0", type=str, help="the embedding dimension")
    parser.add_argument('--analyze', default=False, action="store_true", help="the embedding dimension")
    #parser.add_argument('--num_epoch', type=int, default=10, help="the number of epochs for training")
    parser.add_argument('--threshold', type=float, default=0.5, help="the threshold for the prediction")
    parser.add_argument("--output_dir", default="checkpoints/", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
    args = parser.parse_args()
    return args


args = parse_opt()
device = torch.device('cuda')

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

with open('../preprocessed_data_program/vocab.json') as f:
    vocab = json.load(f)

ivocab = {w: k for k, w in vocab.items()}


def back_to_words(seq):
    return " ".join([ivocab[_.item()] for _ in seq if ivocab[_.item()] != "<PAD>"])


start_time = time.time()
if args.do_train:
    train_examples = get_batch(option='train', data_dir=args.data_dir, vocab=vocab,
                               max_seq_length=args.max_seq_length, cutoff=-1)
    train_data = TensorDataset(*train_examples)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

if args.do_val:
    val_examples = get_batch(option='val', data_dir=args.data_dir, vocab=vocab, max_seq_length=args.max_seq_length)
    val_data = TensorDataset(*val_examples)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

if args.do_test:
    val_examples = get_batch(option='test', data_dir=args.data_dir, vocab=vocab,  max_seq_length=args.max_seq_length)
    val_data = TensorDataset(*val_examples)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

if args.do_simple_test:
    val_examples = get_batch(option='simple_test', data_dir=args.data_dir,
                             vocab=vocab,  max_seq_length=args.max_seq_length)
    val_data = TensorDataset(*val_examples)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

if args.do_complex_test:
    val_examples = get_batch(option='complex_test', data_dir=args.data_dir,
                             vocab=vocab,  max_seq_length=args.max_seq_length)
    val_data = TensorDataset(*val_examples)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

if args.do_small_test:
    val_examples = get_batch(option='small_test', data_dir=args.data_dir,
                             vocab=vocab,  max_seq_length=args.max_seq_length)
    val_data = TensorDataset(*val_examples)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

print("Loading used {} secs".format(time.time() - start_time))

encoder_stat = Encoder(vocab_size=len(vocab), d_word_vec=128, n_layers=args.layer_num, d_model=128, n_head=4)
encoder_prog = Decoder(vocab_size=len(vocab), d_word_vec=128, n_layers=args.layer_num, d_model=128, n_head=4)

encoder_stat.to(device)
encoder_prog.to(device)
# classifier.to(device)


def evaluate(val_dataloader, encoder_stat, encoder_prog):
    mapping = {}
    back_mapping = {}
    all_idexes = set()
    accuracy = 0
    TP, TN, FN, FP = 0, 0, 0, 0
    for val_step, batch in enumerate(val_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, prog_ids, labels, index, true_lab, pred_lab = batch

        enc_stat = encoder_stat(input_ids)
        enc_prog, logits = encoder_prog(prog_ids, input_ids, enc_stat)

        similarity = torch.sigmoid(logits)
        #similarity = torch.sigmoid(classifier(torch.cat([enc_stat, enc_prog], -1)).squeeze())
        similarity = similarity.cpu().data.numpy()
        sim = (similarity > args.threshold).astype('float32')
        labels = labels.cpu().data.numpy()
        index = index.cpu().data.numpy()
        true_lab = true_lab.cpu().data.numpy()
        pred_lab = pred_lab.cpu().data.numpy()

        TP += ((sim == 1) & (labels == 1)).sum()
        TN += ((sim == 0) & (labels == 0)).sum()
        FN += ((sim == 0) & (labels == 1)).sum()
        FP += ((sim == 1) & (labels == 0)).sum()

        if not args.voting:
            for i, s, p, t, inp_id, prog_id in zip(index, similarity, pred_lab, true_lab, input_ids, prog_ids):
                if args.analyze:
                    inp = back_to_words(inp_id)
                    r = back_to_words(prog_id[1:])
                else:
                    inp = None
                    r = None
                if i not in mapping:
                    mapping[i] = [s, numpy.asscalar(p), numpy.asscalar(t), inp, r]
                else:
                    if s > mapping[i][0]:
                        mapping[i] = [s, numpy.asscalar(p), numpy.asscalar(t), inp, r]
        else:
            factor = 2
            for i, s, p, t in zip(index, similarity, pred_lab, true_lab):
                if i not in mapping:
                    if p == 1:
                        mapping[i] = [factor * s, s, t]
                    else:
                        mapping[i] = [-s, s, t]
                else:
                    if p == 1:
                        mapping[i][0] += factor * s
                    else:
                        mapping[i][0] -= s

    precision = TP / (TP + FP + 0.001)
    recall = TP / (TP + FN + 0.001)
    print("TP: {}, FP: {}, FN: {}, TN: {}. precision = {}: recall = {}".format(TP, FP, FN, TN, precision, recall))

    results = []
    if not args.voting:
        success, fail = 0, 0
        for i, line in mapping.items():
            if line[1] == line[2]:
                success += 1
            else:
                fail += 1
            results.append({'pred': line[1], 'gold': line[2], 'fact': line[3], 'program': line[4]})

        print("success = {}, fail = {}, accuracy = {}".format(success, fail, success / (success + fail + 0.001)))
        accuracy = success / (success + fail + 0.001)
    else:
        success, fail = 0, 0
        for i, ent in mapping.items():
            if (ent[0] > 0 and ent[2] == 1) or (ent[0] < 0 and ent[2] == 0):
                success += 1
            else:
                fail += 1
        print("success = {}, fail = {}, accuracy = {}".format(success, fail, success / (success + fail + 0.001)))
        accuracy = success / (success + fail + 0.001)
    if args.analyze:
        if args.do_test or args.do_small_test:
            with open('/tmp/test_eval_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        if args.do_val:
            with open('/tmp/val_eval_results.json', 'w') as f:
                json.dump(results, f, indent=2)
    return precision, recall, accuracy


if args.resume:
    encoder_stat.load_state_dict(torch.load(args.output_dir + "encoder_stat_{}.pt".format(args.id)))
    encoder_prog.load_state_dict(torch.load(args.output_dir + "encoder_prog_{}.pt".format(args.id)))
    #classifier.load_state_dict(torch.load(args.output_dir + "classifier.pt"))
    print("Reloading saved model {}".format(args.output_dir))

if args.do_train:
    loss_func = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss_func.to(device)
    encoder_stat.train()
    encoder_prog.train()
    # classifier.train()
    print("Start Training with {} batches".format(len(train_dataloader)))

    params = chain(encoder_stat.parameters(), encoder_prog.parameters())
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, params),
                                 lr=args.learning_rate, betas=(0.9, 0.98), eps=0.9e-09)
    best_accuracy = 0
    for epoch in range(10):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, prog_ids, labels, index, true_lab, pred_lab = batch

            encoder_stat.zero_grad()
            encoder_prog.zero_grad()
            optimizer.zero_grad()

            enc_stat = encoder_stat(input_ids)
            enc_prog, logits = encoder_prog(prog_ids, input_ids, enc_stat)
            """
			mag_stat = torch.norm(enc_stat, p=2, dim=1)
			mag_prog = torch.norm(enc_prog, p=2, dim=1)
			similarity = (enc_stat * enc_prog).sum(-1) / (mag_stat * mag_prog)

			loss = -torch.mean(torch.log(similarity))
			"""
            loss = loss_func(logits, labels)

            similarity = torch.sigmoid(logits)
            pred = (similarity > args.threshold).float()

            loss.backward()
            optimizer.step()

            if (step + 1) % 20 == 0:
                print("Loss function = {}".format(loss.item()),
                      list(pred[:10].cpu().data.numpy()),
                      list(labels[:10].cpu().data.numpy()))

            if (step + 1) % 200 == 0:
                encoder_stat.eval()
                encoder_prog.eval()
                # classifier.eval()

                precision, recall, accuracy = evaluate(val_dataloader, encoder_stat, encoder_prog)

                if accuracy > best_accuracy:
                    torch.save(encoder_stat.state_dict(), args.output_dir + "encoder_stat_{}.pt".format(args.id))
                    torch.save(encoder_prog.state_dict(), args.output_dir + "encoder_prog_{}.pt".format(args.id))
                    #torch.save(classifier.state_dict(), args.output_dir + "classifier.pt")
                    best_accuracy = accuracy

                encoder_stat.train()
                encoder_prog.train()
                # classifier.train()

if args.do_val or args.do_test or args.do_simple_test or args.do_complex_test or args.do_small_test:
    encoder_stat.eval()
    encoder_prog.eval()
    # classifier.eval()
    precision, recall, accuracy = evaluate(val_dataloader, encoder_stat, encoder_prog)
