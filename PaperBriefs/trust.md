# A Developmental Cognitive Architecture for Trust and Theory of Mind in Humanoid Robots

## Abstract
As artificial systems are starting to be widely deployed in real-world settings, it becomes critical to provide them with the ability to discriminate between different informants and to learn from reliable sources. Moreover, equipping an artificial agent to infer beliefs may improve the collaboration between humans and machines in several ways. In this article, we propose a hybrid cognitive architecture, called Thrive, with the purpose of unifying in a computational model recent discoveries regarding the underlying mechanism involved in trust. The model is based on biological observations that confirmed the role of the midbrain in trial-and-error learning, and on developmental studies that indicate how essential is a theory of mind in order to build empathetic trust. Thrive is build on top of an actor–critic framework that is used to stabilize the weights of two self-organizing maps. A Bayesian network embeds prior knowledge into an intrinsic environment, providing a measure of cost that is used to boostrap learning without an external reward signal. Following a developmental robotics approach, we embodied the model in the iCub humanoid robot and we replicated two psychological experiments. The results are in line with real data, and shed some light on the mechanisms involved in trust-based learning in children and robots.

## Citation
@article{patacchiola2020developmental,
  title={A developmental cognitive architecture for trust and theory of mind in humanoid robots},
  author={Patacchiola, Massimiliano and Cangelosi, Angelo},
  journal={IEEE Transactions on Cybernetics},
  year={2020},
  publisher={IEEE}
}

## Link
https://ieeexplore.ieee.org/iel7/6221036/6352949/09136927.pdf

## Motivations
- AIs and robots will be entering the world and learning from interactions with humans
- Some human sources are unreliable such as the ones the Microsoft chat bot Tay that became a nazi learned from

## Goals
- Create a model of robot which can learn over time which sources are reliable

## Model
- Using an actor critic model based on the brain
- The actor takes in information through sensor and acts on the enviroment, and the critic gets feedback from the enviroment

## For our purposes
- Demonstrates an implementation of a multisectioned brain working together
- Could include sections which primarily act on each other as controls rather than acting on the enviroment
- Important step toward general AI

## Notes
Claim: As artificial intelligent robots enter the world they will learn from interactions with people, they need to be able to determine trust worthy sources, such as how twitter users made Microsoft’s Tay bot racist.

In the mammalian brain, the interaction between ventral and dorsal striatum, modulated by dopamine, is involved in trial-and-error learning. It is often referred to the ventral striatum as the critic and to the dorsal striatum as the actor

They use an actor and a critic, the actor acts on the environment and the critic receives feedback from the environment and talks to the actor
