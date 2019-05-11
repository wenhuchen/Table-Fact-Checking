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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default="train", help="whether to train or test the model")
    parser.add_argument('--emb_dim', type=int, default=128, help="the embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.2, help="the embedding dimension")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume previous run")
    parser.add_argument('--batch_size', type=int, default=256, help="the embedding dimension")
    parser.add_argument('--data_dir', type=str, default='data', help="the embedding dimension")
    parser.add_argument('--max_seq_length', type=int, default=100, help="the embedding dimension")
    parser.add_argument('--layer_num', type=int, default=3, help="the embedding dimension")    
    parser.add_argument('--evaluate_every', type=int, default=5, help="the embedding dimension")
    parser.add_argument("--output_dir", default="checkpoints/", type=str, \
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    args = parser.parse_args()
    return args

args = parse_opt()
device = torch.device('cuda')

with open('../data/vocab.json') as f:
    data = json.load(f)
    s_vocab = data['s_vocab']
    a_vocab = data['a_vocab']

start_time = time.time()
if 'train' in args.option:
	train_examples = get_batch(option='val', data_dir='../data', ready_dir='../READY', \
							   s_vocab=s_vocab, a_vocab=a_vocab)
	train_data = TensorDataset(*train_examples)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

if 'val' in args.option:
	val_examples = get_batch(option='val', data_dir='../data', ready_dir='../READY', \
							s_vocab=s_vocab, a_vocab=a_vocab)
	val_data = TensorDataset(*val_examples)
	val_sampler = RandomSampler(val_data)
	val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

if 'test' in args.option:
	test_examples = get_batch(option='test', data_dir='../data', ready_dir='../READY', \
							s_vocab=s_vocab, a_vocab=a_vocab)
	test_data = TensorDataset(*test_examples)
	test_sampler = RandomSampler(test_data)
	test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size) 

print("Loading used {} secs".format(time.time() - start_time))

encoder = Encoder(vocab_size=len(s_vocab), d_word_vec=128, n_layers=3, d_model=128, n_head=4)
decoder = Decoder(vocab_size=len(a_vocab), d_word_vec=128, n_layers=3, d_model=128, n_head=4)

encoder.to(device)
decoder.to(device)

loss_func = torch.nn.BCEWithLogitsLoss()
loss_func.to(device)

best_acc = 0

encoder.train()
decoder.train()

print("Start Training with {} batches".format(len(train_dataloader)))
if args.resume:
	encoder.load_state_dict(torch.load(args.output_dir + "encoder.pt"))
	encoder.load_state_dict(torch.load(args.output_dir + "decoder.pt"))
	print("Reloading saved model {}".format(args.output_dir))

params = chain(encoder.parameters(), decoder.parameters())
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, params),
							lr=args.learning_rate, betas=(0.9, 0.98), eps=0.9e-09)

for epoch in range(10):
	for step, batch in enumerate(train_dataloader):
		batch = tuple(t.to(device) for t in batch)
		input_ids, prog_ids, labels, gt, pred = batch

		encoder.zero_grad()
		decoder.zero_grad()
		optimizer.zero_grad()

		enc_output = encoder(input_ids)
		logits = decoder(prog_ids, input_ids, enc_output)

		loss = loss_func(logits.squeeze(), labels)

		loss.backward()
		optimizer.step()

		print("Loss function = {}".format(loss.item()))

	torch.save(encoder.state_dict(), args.output_dir + "encoder.pt")
	torch.save(decoder.state_dict(), args.output_dir + "decoder.pt")

