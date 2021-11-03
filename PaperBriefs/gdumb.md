# GDumb: A Simple Approach that Questions Our Progress in Continual Learning

## Abstract
We discuss a general formulation for the Continual Learning (CL) problem for classification—a learning task where a stream provides samples to a learner and the goal of the learner, depending on the samples it receives, is to continually upgrade its knowledge about the old classes and learn new ones. Our formulation takes inspiration from the open-set recognition problem where test scenarios do not necessarily belong to the training distribution. We also discuss various quirks and assumptions encoded in recently proposed approaches for CL. We argue that some oversimplify the problem to an extent that leaves it with very little practical importance, and makes it extremely easy to perform well on. To validate this, we propose GDumb that (1) greedily stores samples in memory as they come and; (2) at test time, trains a model from scratch using samples only in the memory. We show that even though GDumb is not specifically designed for CL problems, it obtains state-of-the-art accuracies (often with large margins) in almost all the experiments when compared to a multitude of recently proposed algorithms. Surprisingly, it outperforms approaches in CL formulations for which they were specifically designed. This, we believe, raises concerns regarding our progress in CL for classification. Overall, we hope our formulation, characterizations and discussions will help in designing realistically useful CL algorithms, and GDumb will serve as a strong contender for the same.

## Citation
@inproceedings{prabhu2020gdumb,
  title={Gdumb: A simple approach that questions our progress in continual learning},
  author={Prabhu, Ameya and Torr, Philip HS and Dokania, Puneet K},
  booktitle={European conference on computer vision},
  pages={524--540},
  year={2020},
  organization={Springer}
}

## Link
https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf

## Motivations

## Goals

## Model

## For our purposes

## Notes

