# A Conceptual Bio-Inspired Framework for the Evolution of Artificial General Intelligence

## Abstract
In this work, a conceptual bio-inspired parallel and distributed learning framework for the emergence of general intelligence is proposed, where agents evolve through environmental rewards and learn throughout their lifetime without supervision, i.e., self-learning through embodiment. The chosen control mechanism for agents is a biologically plausible neuron model based on spiking neural networks. Network topologies become more complex through evolution, i.e., the topology is not fixed, while the synaptic weights of the networks cannot be inherited, i.e., newborn brains are not trained and have no innate knowledge of the environment. What is subject to the evolutionary process is the network topology, the type of neurons, and the type of learning. This process ensures that controllers that are passed through the generations have the intrinsic ability to learn and adapt during their lifetime in mutable environments. We envision that the described approach may lead to the emergence of the simplest form of artificial general intelligence.

general intelligence.
## Citation
@article{pontes2019conceptual,
  title={A Conceptual Bio-Inspired Framework for the Evolution of Artificial General Intelligence},
  author={Pontes-Filho, Sidney and Nichele, Stefano},
  journal={arXiv preprint arXiv:1903.10410},
  year={2019}
}

## Link
https://arxiv.org/pdf/1903.10410

## Motivations
- The advancement of AI towards general intelligence

## Goals
- Propose a general framwork for AI which could lead further towrds gen AI

## Model
- Unsupervised learning in a enviroment 
- Embodiment
- Spiking neurons
- Evolution passes down the structure of the networks, not weights
- Mutable enviroment
- Each neuron can learn in different ways ex: hebbian, anti-hebbian

## For our purposes
- Builds out a nice framework with a lot of details we are interested in incorporating
- Suggests using a game or prexisting mutable enviroment to train, would be cool to put AIs in a game

## Notes
They have agents learn unsupervised and use embodiment with an environment. Network topology is evolved and passed down, weights are not passed down. Uses spiking neurons
I think it would be cool to have a combination of evolution and development where the genotype dictates exactly how the network will grow based on stimuli in the environment.
 I think that evolution is important to the problem I had earlier, I was thinking how can you teach the network to like some stimuli and not others, I think development and evolution are needed in combination to produce what we are looking for.

The paper talks about PolyWorld!
Some researchers used quake 3 as an environment and their ais were able to outperform human levels. Maybe we could exploit an existing game environment for our project and it would give us a cool goal to achieve.
Neural MMO could help with this!
And if we could interact in the environment that would be awesome
They propose self-learning through embodiment, mutable environment
They have a binary input and the agent must choose to eat or avoid
I think it would be so cool to have an evolutionary genotype that could include the learning rules and body development rules, and then also have a system for energy and health so they need to eat food and avoid damage. It can be modular like originally proposed, so that we can incorporate new and more complex rules and environments. Also the energy system can give the brain a reason to prune neurons  to be more energy efficient
Each neuron in their model can have different plasticity rules like: asymmetric Hebbian, symmetric Hebbian, asym-metric anti-Hebbian, symmetric anti-Hebbian

