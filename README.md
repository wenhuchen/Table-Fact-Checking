# Introduction
We introduce a large-scale dataset called [**TabFact**](https://tabfact.github.io/), which consists of 118,439 manually annotated statements with regard to 16,621 Wikipedia tables, their relations are classified as *ENTAILED* and *REFUTED*. The full paper is "[TabFact: A Large-scale Dataset for Table-based Fact Verification
](https://arxiv.org/pdf/1909.02164.pdf)"

<p align="center">
<img src="resource/example.png" width="700">
</p>
The table-based fact verification is the first dataset to perform fact verification on strctured data, which involves mixed reasoning in both symbolic and linguistic form. Therefore, we propose two models, namely Table-BERT and the Latent Program Algorithm to tackle this task.

- The brief architecture of Latent Program Algorithm (LPA) looks like following:
<p align="center">
<img src="resource/program.jpg" width="700">
</p>

- The brief architecture of Table-BERT looks like following:
<p align="center">
<img src="resource/BERT.jpg" width="700">
</p>

## Explore the data
We design an interface for you to browse and eplore the dataset in https://tabfact.github.io/explore.html

## Requirements
- Python 3.5
- Pytorch 1.0
- Ujson 1.35
- Pytorch 1.0+
- Pytorch_Pretrained_Bert 0.6.2 (Huggingface Implementation)
- Pandas

## Data Preprocessing
The folder "collected_data" contains the raw data collected directly from Mechnical Turker, all the text are lower-cased, containing foreign characters in some tables. There are two files, the r1 file is collected in the first round (simple channel), which contains sentences involving less reasoning. The r2 file is collected in the second round (complex channel), which involves more complex multi-hop reasoning. These two files in total contains roughly 110K statements, the positive and negative satements are balanced. We demonstrate the data format as follows:
  ```
  Table-id: {
  [
  Statement 1,
  Statement 2,
  ...
  ],
  [
  Label 1,
  Label 2,
  ...
  ],
  Table Caption
  }
  ```
1. General Tokenization and Entity Matching
    - tokenized_data: This folder contains the data after tokenization with preprocess_data.py by:
      ```
      cd code/
      python preprocess_data.py
      ```
      this script is mainly used for feature-based entity linking, the entities in the statements are linked to the longest text span in the table cell. The result file is tokenized_data/full_cleaned.json, which has a data format like:
      ```
      Table-id: {
      [
      Statement 1: xxxxx #xxx;idx1,idx2# xxx.
      Statement 2: xx xxx #xxx;idx1,idx2# xxx.
      ...
      ],
      [
      Label 1,
      Label 2,
      ...
      ],
      Table Caption
      }
      ```
      The enclosed snippet #xxx;idx1,idx2# denotes that the word "xxx" is linked to the entity residing in idx1-th row and idx2-th column of table "Table-id.csv", if idx1=-1, it links to the table caption. The entity linking step is essential for performing  the following program search algorithm.

2. Tokenization For Latent Program Algorithm
    - preprocessed_data_program: This folder contains the preprocessed.json, which is obtained by:
      ```
      cd code/
      python run.py
      ```
      this script is mainly used to perform cache (string, number) initialization, the result file looks like:
      ```
      [
        [
        Table-id,
        Statement: xxx #xxx;idx1,idx2# (after entity linking),
        Pos-Tagging information,
        Statement with place-holder,
        [linked string entity],
        [linked number entity],
        [linked string header],
        [linked number header],
        Statement-id,
        Label
        ],
      ]
      ```
      This file is directly fed into run.py to search for program candidates using dynamic programming, which also contains the tsv files neccessary for the program ranking algorithm.
    - all_programs: this folder contains the searched intermediate results for different statements, we save the results in different files for different statements, the format of intermediate program results looks like:
      ```
      [
        csv_file,
        statement,
        placeholder-text,
        label,
        [
          program1,
          program2,
          ...
        ]
      ]
      ```
2. Tokenization for Table-BERT
```
  cd code/
  python preprocess_BERT.py --scan horizontal
  python preprocess_BERT.py --scan vertical
```

## Latent Program Algorithm
1. Downloading the preprocessed data for LPA
Here we provide the data we obtained after preprocessing through the above pipeline, you can download that by running

```
  sh get_data.sh
```

2. Training the ranking model
Once we have all the training and evaluating data in folder "preprocessed_data_program", we can simply run the following command to evaluate the fact verification accuracy as follows:

```
  cd code/
  python model.py --do_train --do_val
```
3. Evaluating the ranking model
We have put our pre-trained model in code/checkpoints/, the model can reproduce the exact number reported in the paper:
```
  cd code/
  python model.py --do_test --resume
  python model.py --do_simple --resume
  python model.py --do_complex --resume
```
## Table-BERT
1. Training the verification model
```
  cd code/
  python run_BERT.py --do_train [--do_eval] --scan [horizontal, vertical] --fact [first/second]
```
2. Evaluating the verification model
```
  cd code/
  python run_BERT.py --do_eval --scan [horizontal, vertical] --fact [first/second] --load_dir YOUR_TRAINED_MODEL --eval_batch_size N

```

## If you find this code useful, please cite the paper
```
@inproceedings{2019TabFactA,
  title={TabFact : A Large-scale Dataset for Table-based Fact Verification},
  author={Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou and William Yang Wang},
  year={2019}
}
```

## Q&A
If you encounter any problem, please either directly contact the first author or leave an issue in the github repo.

