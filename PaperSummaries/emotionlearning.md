Uses several emotional parameters, accounts for perdicted reward and actual reward


Hebian learning causes weights to grow very large, hard to search a large network due to large dimensionality

Oja learning uses a forgetting term to restrain unlimited growth of weights. The forgetting term is proportional to the principal component of the input-unit correlation matrix. Reduces the dimensionality
Requires an objective function with both error term and some regulation terms

As the reward increases over time, the agent can counter dispersive effects and keep a stable performance, and the higher the valence of agent will become. We thus define the valence of agent as a decreasing function of the estimation of entropy of the rewards over time.

a hypothesis has been proposed: dopamine and serotonin mainly influences the reward prediction error, noradrenaline affects the randomness in action selection, and acetylcholine controls the speed of memory update.



Flow of generation and modulation

1. Inspired by dopaminergic modulation in the cerebral cortex, we suggest that emotional valence is able to change the reward prediction error by adjusting the baseline. From the perspective of computation, emotion can adaptively adjust the filter factor so that estimating upcoming reward depends on either more recent rewards or more past experience. Here, the filter factor is simply defined as a decreasing function of valence


2. Emotion can modulate the parameter of learning rate ηn . Neurobiological studies [54], [55] have shown that acetylcholine is able to modulate the synaptic plasticity in the hippocampus and the cerebral cortex, and then influences the learning process and memories. Here, emotion modulates cognitive and behavioral learning system by controlling the learning rate, which is simply governed by

3. Emotion can control the randomness in action selection through adjusting the variance of the node-perturbation noise. In the viewpoint of neurobiology, noradrenaline has the function of balancing the wide exploration and focused execution. Meanwhile, in related psychological studies [56], [57], on the problems involving risky choices, some specific negative emotions tend to increase risk-taking choices, whereas others act risk-averse. Hence, we propose that the neural system tends to more exploration when valence is low, conversely more exploitation when valence is high, which can be implemented by adjusting the variance of noise


Training:

Get output

Compute rewards

compute emotional valence

adjust parameters with emotional valence

adjust weights


