# AR2 Project

This repo provides the code of AR2 with passage sieve algorithm. We built the repo on top of [AR2](https://github.com/microsoft/AR2/tree/main/AR2)  and [SimANS](https://github.com/microsoft/SimXNS/tree/main/SimANS). And the additional files of our repo are as follows:
 -  AR2/sieve/loss.py
 -  AR2/wiki/dpr.py
 -  AR2/wiki/negative_train.py
 -  AR2/co_training/dpr.py
 -  AR2/co_trianing/negative_train_ms.py

### Data preparation
Basically, the training of AR2 includes warm-up training and AR2 training two parts; We add an another data preprocessing, i.e. passage sieve, into the AR2 training part.

The required data and warm-up checkpoints for training can be downloaded accoring to the repo of [SimANS](https://github.com/microsoft/SimXNS/tree/main/SimANS) 

### Start training
One can directly run the following scripts under the AR2 folder to reproduce the experiments results:
```
bash ../train_nq.sh # for Natural-QA dataset
bash ../train_tq.sh # for Trivia-QA dataset
bash ../train_ms.sh # for MS-pas dataset
```

### Final checkpoints
The final checkponits of all datasets will be released once the paper are accepted.