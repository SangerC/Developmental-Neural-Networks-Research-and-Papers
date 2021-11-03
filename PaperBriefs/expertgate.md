# Expert Gate: Lifelong Learning with a Network of Experts

## Abstract
In this paper we introduce a model of lifelong learning, based on a Network of Experts. New tasks / experts are learned and added to the model sequentially, building on what was learned before. To ensure scalability of this process, data from previous tasks cannot be stored and hence is not available when learning a new task. A critical issue in such context, not addressed in the literature so far, relates to the decision which expert to deploy at test time. We introduce a set of gating autoencoders that learn a representation for the task at hand, and, at test time, automatically forward the test sample to the relevant expert. This also brings memory efficiency as only one expert network has to be loaded into memory at any given time. Further, the auto encoders inherently capture the relatedness of one task to another, based on which the most relevant prior model to be used for training a new expert, with fine-tuning or learning-without-forgetting, can be selected. We evaluate our method on image classification and video prediction problems.

## Citation
@inproceedings{aljundi2017expert,
  title={Expert gate: Lifelong learning with a network of experts},
  author={Aljundi, Rahaf and Chakravarty, Punarjay and Tuytelaars, Tinne},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3366--3375},
  year={2017}
}

## Link
http://openaccess.thecvf.com/content_cvpr_2017/papers/Aljundi_Expert_Gate_Lifelong_CVPR_2017_paper.pdf

## Motivations
- The continuous problem of continual learning, networks learn a task and in order to learn another must forget it

## Goals
- Proposes a novel approach to learning multiple tasks with a system of networks working together

## Model
- Gate network determines what the task is and routes it to a network which can solve it
- If a task has not been done before it creates a new network starting with the network of the most similar task

## For our purposes
- More ideas of multi part networks/brains
- Find ways to integrate experts and could use a similar model with a developmental approach for the experts

## Notes
Expert-Gate: Uses a network to detect what problem it is and routes it to an appropriate expert or creates a new one, it starts an expert for a new task at the most relevant expert
