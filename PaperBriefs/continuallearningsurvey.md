# A continual learning survey: Defying forgetting in classification tasks

## Abstract
Artificial neural networks thrive in solving the classification problem for a particular rigid task, acquiring knowledge through generalized learning behaviour from a distinct training phase. The resulting network resembles a static entity of knowledge, with endeavours to extend this knowledge without targeting the original task resulting in a catastrophic forgetting. Continual learning shifts this paradigm towards networks that can continually accumulate knowledge over different tasks without the need to retrain from scratch. We focus on task incremental classification, where tasks arrive sequentially and are delineated by clear boundaries. Our main contributions concern (1) a taxonomy and extensive overview of the state-of-the-art; (2) a novel framework to continually determine the stability-plasticity trade-off of the continual learner; (3) a comprehensive experimental comparison of 11 state-of-the-art continual learning methods and 4 baselines. We empirically scrutinize method strengths and weaknesses on three benchmarks, considering Tiny Imagenet and large-scale unbalanced iNaturalist and a sequence of recognition datasets. We study the influence of model capacity, weight decay and dropout regularization, and the order in which the tasks are presented, and qualitatively compare methods in terms of required memory, computation time and storage.

## Citation
@ARTICLE{9349197,
	author={Delange, Matthias and Aljundi, Rahaf and Masana, Marc and Parisot, Sarah and Jia, Xu and Leonardis, Ales and Slabaugh, Greg and Tuytelaars, Tinne},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A continual learning survey: Defying forgetting in classification tasks}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3057446}
}

## Link
https://ieeexplore.ieee.org/abstract/document/9349197

## Motivations
- The continuous problem of continual learning, networks learn a task and in order to learn another must forget it
- To survey the methods and state of the art

## Goals
- To survey the methods and state of the art
- Give researchers a starting point and base knowledge on the topic

## Model
- There are 3 main categories of continual learning methods
- Replay: take samples of inputs and replay them while training a new task to keep in that direction
- Regularization: Data based: use previous model output as soft labels for previous tasks Data, Prior-focused: find the importance of parameters and make the most important ones hard to change when learning new data, Prior-focused: find the importance of parameters and make the most important ones hard to change when learning new data
- Parameter isolation: dedicate different parameters to different tasks, like expert gate creating a new network for each task

## For our purposes
- Show many ways in which we can make a network learn multiple  tasks without forgetting and we can apply to a developmental approach
- Some methods like expert gate are somewhat developmental themselves

## Notes
Three methods: 
• Replay methods
• Regularization-based methods
• Parameter isolation methods

Replay:
This line of work stores samples in raw format or generates pseudo-samples with a generative model. These previous task samples are replayed while learning a new task to alleviate forgetting.
    Rehearsal: retrain on limited subset of samples while training new task, bounded by 
effectiveness of joint training

Constrained Optimization: constrain new task updates to not interfere with the previous 
tasks, projecting the estimated gradient direction on the feasible region outlined by 
previous gradients

pseudo rehearsal: the output of the model given random inputs creates your samples

These are very traditional neural networky, not what we are super interested in

Regularization-based methods: Doesn’t store inputs, an extra regularization term is introduced to the loss function.
    Data-focused: use the previous task model outputs given new task input images
    Prior-focused methods: estimate a distribution over model parameters, used as a prior when learning from new data

Parameter isolation methods: uses different model parameters to each task, build a new branch for each task without constraint on network size.
need to change what it is based on task typically simply has a option for which task, but cannot have one head. Expert Gate gets around this through an auto-encoder gate. I think this is what I was talking about maybe!!! having a network part at the front that detects what the problem is.

Then it goes to compare all this.
