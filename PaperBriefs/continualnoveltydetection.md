# Continual Novelty Detection

## Abstract
Novelty Detection methods identify samples that are not representative of a model’s training set thereby flagging misleading predictions and bringing a greater flexibility and transparency at deployment time. However, research in this area has only considered Novelty Detection in the offline setting. Recently, there has been a growing realization in the computer vision community that applications demand a more flexible framework - Continual Learning where new batches of data representing new domains, new classes or new tasks become available at different points in time. In this setting, Novelty Detection becomes more important, interesting and challenging. This work identifies the crucial link between the two problems and investigates the Novelty Detection problem under the Continual Learning setting. We formulate the Continual Novelty Detection problem and present a benchmark, where we compare several Novelty Detection methods under different Continual Learning settings. We show that Continual Learning affects the behaviour of novelty detection algorithms , while novelty detection can pinpoint insights in the behaviour of a continual learner. We further propose baselines and discuss possible research directions. We believe that the coupling of the two problems is a promising direction to bring vision models into practice.

## Citation
@article{aljundi2021continual,
  title={Continual Novelty Detection},
  author={Aljundi, Rahaf and Reino, Daniel Olmeda and Chumerin, Nikolay and Turner, Richard E},
  journal={arXiv preprint arXiv:2106.12964},
  year={2021}
}

## Link
https://arxiv.org/pdf/2106.12964.pdf

## Motivations
- Research in novelty detection has typically been done in an offline setting (all the data is present)
- In online continual learning new tasks and sets of data become available at different times

## Goals
- Show the link between online continual learning and novelty detection
- Compare under different continual learning settings
- Show that continual learning affects the behaviour of novelty detection algorithms

## Model
- Batches of labelled data arrive sequentially at different points in time, each new batch corresponds to a new group of categories that could form a separate task
- ND(Mt) should be able to after each stage how familiar a given input is
- Novel inputs can be gathered and annotated
- After each stage produce 3 sets: IN: samples correctly placed in a previous set, OUT: Samples of unknown, FORG: samples that were known and predicted correctly, but were predicted incorrectly now
- Multihead and shared head: different tasks have different networks vs same network and use something like parameter distributions
- ND methods: Softmax, ODIN, Mahalanobis, VAE

## For our purposes
- Shows results of several recent methods for novelty detection, and how they interact with CL!

## Notes

