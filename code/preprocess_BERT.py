from __future__ import division
import random
import sys
import io
import os
import logging
import re
import pandas as pd
import ujson as json
import os.path as op
from tqdm import tqdm
from collections import Counter, OrderedDict
import argparse


program = os.path.basename(sys.argv[0])
L = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
L.info("Running %s" % ' '.join(sys.argv))

entity_linking_pattern = re.compile('#.*?;-*[0-9]+,(-*[0-9]+)#')
fact_pattern = re.compile('#(.*?);-*[0-9]+,-*[0-9]+#')
unk_pattern = re.compile('#([^#]+);-1,-1#')
TSV_DELIM = "\t"
TBL_DELIM = " ; "


def join_unicode(delim, entries):
    #entries = [_.decode('utf8') for _ in entries]
    return delim.join(entries)


def parse_fact(fact):
    fact = re.sub(unk_pattern, '[UNK]', fact)
    chunks = re.split(fact_pattern, fact)
    output = ' '.join([x.strip() for x in chunks if len(x.strip()) > 0])
    return output


def process_file(data_dir, shuffle=False):

    all_csv_dir = op.join(data_dir, "data/all_csv")
    all_data = op.join(data_dir, "tokenized_data/full_cleaned.json")

    examples = []
    with io.open(all_data, 'r', encoding='utf8') as fin:
        dataset = json.load(fin)

        for idx, (fname, sample) in tqdm(enumerate(dataset.items())):

            try:
                table = pd.read_csv(op.join(all_csv_dir, fname), delimiter='#')

                # facts: list of strings
                facts = sample[0]
                # labels: list of ints
                labels = sample[1]
                assert all([x in [0, 1] for x in labels])
                assert len(facts) == len(labels)

                # types: list of table column strings
                types = [str(x) for x in table.columns.values.tolist()]

                # columns: {type: list of cell phrases in this column}
                columns = OrderedDict()
                for t in types:
                    # np array of cells in the one-column table (dataframe) --> list
                    one_column = [str(x) for x in table[t].to_numpy().tolist()]
                    columns[t] = one_column

                # pack into one example
                example = {
                    "csv": fname,
                    "columns": columns,
                    "facts": facts,
                    "labels": labels
                }
                examples.append(example)

            except:
                print("{} is misformated".format(fname))

        if shuffle:
            random.shuffle(examples)

        print("{} samples in total".format(len(examples)))

    return examples


def convert_to_tsv(out_file, examples, dataset_type, meta, scan):

    L.info("Processing {} examples...".format(dataset_type))
    total = 0

    unk = 0
    len_total = 0
    empty_table = 0
    with io.open(out_file, 'w', encoding='utf-8') as fout:
        for example in tqdm(examples):
            assert len(example['facts']) == len(example['labels'])
            for fact, label in zip(example['facts'], example['labels']):
                # use entity linking info to retain relevant columns
                useful_column_nums = [int(x) for x in re.findall(entity_linking_pattern, fact) if not x == '-1']
                useful_column_nums = dict.fromkeys(useful_column_nums)
                remaining_table = OrderedDict()
                for idx, (column_type, column_cells) in enumerate(example['columns'].items()):
                    if idx in useful_column_nums:
                        column_type = '_'.join(column_type.split())
                        remaining_table[column_type] = column_cells

                fact_clean = parse_fact(fact)
                if len(remaining_table) > 0:
                    table_cells, table_feats = [], []

                    len_total += 1
                    if scan == 'vertical':
                        for column_type, column_cells in remaining_table.items():
                            column_type = ' '.join(column_type.split('_'))
                            table_cells.extend([column_type, 'are :'])
                            this_column = []
                            for idx, c in enumerate(column_cells):
                                this_column.append("row {} is {}".format(idx + 1, c))
                            this_column = join_unicode(TBL_DELIM, this_column)
                            table_cells.append(this_column)
                            table_cells.append('.')
                            table_feats.append(column_type)
                    else:
                        # stupid but to reserve order
                        table_column_names, table_column_cells = [], []
                        for column_type, column_cells in remaining_table.items():
                            column_type = ' '.join(column_type.split('_'))
                            table_feats.append(column_type)
                            table_column_names.append(column_type)
                            table_column_cells.append(column_cells)
                        for idx, row in enumerate(zip(*table_column_cells)):
                            table_cells.append('row {} is :'.format(idx + 1))
                            this_row = []
                            for col, tk in zip(table_column_names, row):
                                this_row.append('{} is {}'.format(col, tk))
                            this_row = join_unicode(TBL_DELIM, this_row)
                            table_cells.append(this_row)
                            table_cells.append('.')

                    table_str = ' '.join(table_cells)
                    out_items = [example['csv'],
                                 str(len(table_feats)),
                                 ' '.join([str(x) for x in table_feats]),
                                 table_str,
                                 fact_clean,
                                 str(label)]

                    out_items = TSV_DELIM.join(out_items)
                    total += 1
                    fout.write(out_items + "\n")
                else:
                    if dataset_type != 'train':
                        table_feats = ['[UNK]']
                        table_cells = ['[UNK]']
                        table_str = ' '.join(table_cells)
                        out_items = [example['csv'],
                                     str(len(table_feats)),
                                     ' '.join([str(x) for x in table_feats]),
                                     table_str,
                                     fact_clean,
                                     str(label)]

                        out_items = TSV_DELIM.join(out_items)
                        fout.write(out_items + "\n")
                        total += 1
                    empty_table += 1
    print("Built {} instances of features in total, {}/{}={}% unseen column types, {} empty tables"
          .format(total, unk, len_total, "{0:.2f}".format(unk * 100 / len_total), empty_table))
    meta["{}_total".format(dataset_type)] = total

    return meta


def split_dataset(data_dir, all_examples):
    total_size = len(all_examples)
    L.info("split {} tables into train dev test ...".format(total_size))

    data_dir = op.join(data_dir, "data/")
    csv_id_lkt = {}
    for x in ['train', 'val', 'test', 'small_test', 'simple_test', 'complex_test']:
        id_file = op.join(data_dir, "{}_id.json".format(x))
        with io.open(id_file, 'r', encoding='utf-8') as fin:
            csv_id_lkt[x] = dict.fromkeys(json.load(fin), True)

    trainset, validset, testset, small_test, simple_test, complex_test = [], [], [], [], [], []
    for sample in all_examples:
        if sample['csv'] in csv_id_lkt['small_test']:
            small_test.append(sample)
        if sample['csv'] in csv_id_lkt['simple_test']:
            simple_test.append(sample)
        if sample['csv'] in csv_id_lkt['complex_test']:
            complex_test.append(sample)

        if sample['csv'] in csv_id_lkt['train']:
            trainset.append(sample)
        elif sample['csv'] in csv_id_lkt['val']:
            validset.append(sample)
        elif sample['csv'] in csv_id_lkt['test']:
            testset.append(sample)
        else:
            print('{} is NOT used'.format(sample['csv']))

    return trainset, validset, testset, small_test, simple_test, complex_test


def save(filename, obj, message=None, beautify=False):
    assert message is not None
    print("Saving {} ...".format(message))
    with io.open(filename, "a") as fh:
        if beautify:
            json.dump(obj, fh, sort_keys=True, indent=4)
        else:
            json.dump(obj, fh)


def mkdir_p(path1, path2=None):
    if path2 is not None:
        path1 = os.path.join(path1, path2)
    if not os.path.exists(path1):
        os.mkdir(path1)
    return path1


def count_types(dataset):
    type_cnt = []
    for example in dataset:
        for name in example['columns'].keys():
            type_cnt.append('_'.join(name.split()))
    return type_cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default='../',
                        help="The path of TabFact folder")
    parser.add_argument("--output_dir",
                        type=str,
                        default='../processed_datasets',
                        help="The path to save output tsv files")
    parser.add_argument("--scan",
                        default="horizontal",
                        choices=["vertical", "horizontal"],
                        type=str,
                        help="The direction of sequentializing table cells.")
    args = parser.parse_args()

    root_dir = mkdir_p(args.output_dir)
    data_save_dir = mkdir_p(root_dir, "tsv_data_{}".format(args.scan))
    train_tsv = os.path.join(data_save_dir, "train.tsv")
    dev_tsv = os.path.join(data_save_dir, "dev.tsv")
    test_tsv = os.path.join(data_save_dir, "test.tsv")
    small_test_tsv = os.path.join(data_save_dir, "small_test.tsv")
    simple_test_tsv = os.path.join(data_save_dir, "simple_test.tsv")
    complex_test_tsv = os.path.join(data_save_dir, "complex_test.tsv")
    meta_file = os.path.join(data_save_dir, "meta.json")
    type2idx_file = os.path.join(data_save_dir, "type2idx.json")
    idx2type_file = os.path.join(data_save_dir, "idx2type.json")

    L.info("process file ...")
    all_examples = process_file(args.data_dir)

    L.info("spliting datasets ...")
    trainset, devset, testset, small_test, simple_test, complex_test = split_dataset(args.data_dir, all_examples)

    L.info("build tsv datasets ...")
    meta = {}
    meta = convert_to_tsv(train_tsv, trainset, "train", meta, args.scan)
    meta = convert_to_tsv(dev_tsv, devset, "dev", meta, args.scan)
    meta = convert_to_tsv(test_tsv, testset, "test", meta, args.scan)
    meta = convert_to_tsv(small_test_tsv, small_test, "small_test", meta, args.scan)
    meta = convert_to_tsv(simple_test_tsv, simple_test, "simple_test", meta, args.scan)
    meta = convert_to_tsv(complex_test_tsv, complex_test, "complex_test", meta, args.scan)
    save(meta_file, meta, message="meta")
