# SR-GNN

## Paper data and code

This is the code for our AAAI 2019 Paper: [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855). We implemented our model in both **Tensorflow** and **Pytorch**.

Here are two datasets we used in our paper. After download the datasets, you can put them in the folder `datasets/`:

- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup>

There is a small dataset `sample` included in the folder `datasets/`, which could be used to test the correctness of the code.

## Usage

You need to run the file  `datasets/preprocess.py` first to preprocess the data.

​    For example: `cd datasets; python preprocess.py --dataset=sample`

```bash
usage: preprocess.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  dataset name: diginetica/yoochoose/sample
```

Then you can run the file `pytorch_code/main.py` or `tensorflow_code/main.py` to train the model.

​    For example: `cd pytorch_code; python main.py --dataset=sample`

You can add the suffix `--nonhybrid` to use the global preference of a session graph to recommend instead of the hybrid preference.

You can also change other parameters according to the usage:

```bash
usage: main.py [-h] [--dataset DATASET] [--batchSize BATCHSIZE]
               [--hiddenSize HIDDENSIZE] [--epoch EPOCH] [--lr LR]
               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2]
               [--step STEP] [--patience PATIENCE] [--nonhybrid]
               [--validation] [--valid_portion VALID_PORTION]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name:
                        diginetica/yoochoose1_4/yoochoose1_64/sample
  --batchSize BATCHSIZE
                        input batch size
  --hiddenSize HIDDENSIZE
                        hidden state size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of epochs after which the learning rate
                        decay
  --l2 L2               l2 penalty
  --step STEP           gnn propogation steps
  --patience PATIENCE   the number of epoch to wait before early stop
  --nonhybrid           only use the global preference to predict
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion of training set as validation set
```

## Requirements

- Python 3
- PyTorch 0.4.0 or Tensorflow 1.9.0
