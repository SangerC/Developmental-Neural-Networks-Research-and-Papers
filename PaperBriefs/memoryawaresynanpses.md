# Memory Aware Synapses: Learning what (not) to forget

## Abstract
Humans can learn in a continuous manner. Old rarely utilized knowledge can be overwritten by new incoming information while important, frequently used knowledge is prevented from being erased. In artificial learning systems, lifelong learning so far has focused mainly on accumulating knowledge over tasks and overcoming catastrophic forgetting. In this paper, we argue that, given the limited model capacity and the unlimited new information to be learned, knowledge has to be preserved or erased selectively. Inspired by neuroplasticity, we propose a novel approach for lifelong learning, coined Memory Aware Synapses (MAS). It computes the importance of the parameters of a neural network in an unsupervised and online manner. Given a new sample which is fed to the network, MAS accumulates an importance measure for each parameter of the network, based on how sensitive the predicted output function is to a change in this parameter. When learning a new task, changes to important parameters can then be penalized, effectively preventing important knowledge related to previous tasks from being overwritten. Further, we show an interesting connection between a local version of our method and Hebb’s rule, which is a model for the learning process in the brain. We test our method on a sequence of object recognition tasks and on the challenging problem of learning an embedding for predicting <subject, predicate, object> triplets. We show state-of-the-art performance and, for the first time, the ability to adapt the importance of the parameters based on unlabeled data towards what the network needs (not) to forget, which may vary depending on test conditions.

## Citation
@inproceedings{aljundi2018memory,
  title={Memory aware synapses: Learning what (not) to forget},
  author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={139--154},
  year={2018}
}

## Link
http://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf

## Motivations
- The continuous problem of continual learning, networks learn a task and in order to learn another must forget it

## Goals
- Proposes a novel approach to alleviate forgetting when learning new tasks with one network

## Model
- Considers the current approximation of the function F
- Determines how important each weight is to maintain that approximation
- Allows the new task to modify weights, but weights less important to F move more freely
- Cons: Must separate into distinct tasks

## For our purposes
- Could be interesting to implement with developing new nodes as well
- Could use as a pruning and new growth method
- Works building off of it despite being relatively new, newer work by same researches which eliminates separation on tasks :O
- Relates continual learning and hebbian learning

## Notes

