# passage-sieve
This is the official repo of the AAAI2024 paper Mitigating the Impact of False Negatives in Dense Retrieval with Contrastive Confidence Regularization

This branch contains the code which can reproduce our reported results in the paper. 

## AR2 Project

This repo provides the code of AR2 with the passage sieve algorithm. We built the repo on top of [AR2](https://github.com/microsoft/AR2/tree/main/AR2)  and [SimANS](https://github.com/microsoft/SimXNS/tree/main/SimANS). The additional files of our repo are as follows:
 -  AR2/sieve/loss.py
 -  AR2/wiki/dpr.py
 -  AR2/wiki/negative_train.py
 -  AR2/co_training/dpr.py
 -  AR2/co_trianing/negative_train_ms.py

### Data Preparation
The training of AR2 includes two parts: warm-up training and AR2 training; We add another data preprocessing, i.e. passage sieve, into the AR2 training part.

The required data and warm-up checkpoints for training can be downloaded according to the repo of [SimANS](https://github.com/microsoft/SimXNS/tree/main/SimANS) 

### Start training
One can directly run the following scripts under the AR2 folder to reproduce the results of the experiment:
```
cd ./AR2+passage_sieve/AR2
bash ../train_nq.sh # for Natural-QA dataset
bash ../train_tq.sh # for Trivia-QA dataset
bash ../train_ms.sh # for MS-pas dataset
```

### Final checkpoints
The final checkpoints will be released soon.

**NEWS! 2024/1/6** \
NQ checkpoint has been released on [huggingface](https://huggingface.co/wangsky/AR2-sieved-NQ). You can download the model and use it now.  You can check AR2/wiki/co_training_wiki_generate.py for an example code of using the model to encode passages and calculate metrics.

