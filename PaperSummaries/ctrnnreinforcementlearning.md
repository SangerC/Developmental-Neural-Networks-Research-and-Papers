# Papers on Reinforcement Learning CTRNNs

## Reinforcement Learning Algorithm with CTRNN in Continuous Action Space

link: https://www.researchgate.net/profile/Tetsuya-Ogata/publication/221139921_Reinforcement_Learning_Algorithm_with_CTRNN_in_Continuous_Action_Space/links/02bfe510a300d1dfd2000000/Reinforcement-Learning-Algorithm-with-CTRNN-in-Continuous-Action-Space.pdf

### Abstract
Abstract. There are some difficulties in applying traditional reinforce-
ment learning algorithms to motion control tasks of robot. Because most
algorithms are concerned with discrete actions and based on the assump-
tion of complete observability of the state. This paper deals with these
two problems by combining the reinforcement learning algorithm and
CTRNN learning algorithm. We carried out an experiment on the pen-
dulum swing-up task without rotational speed information. It is shown
that the information about the rotational speed, which is considered as
a hidden state, is estimated and encoded on the activation of a context
neuron. As a result, this task is accomplished in several hundred trials
using the proposed algorithm.

### Notes
Focused on movement of robots can be similar but also different to movment of biological creatures

Early paper on using reinforcement learning with CTRNNs

Proposes a learning method, must read more throughly to compare with summer research method

## Reinforcement learning of a continuous motor sequence with hidden states

link: https://d1wqtxts1xzle7.cloudfront.net/47150773/Reinforcement_learning_of_a_continuous_m20160710-27829-1vtsoq0.pdf?1468220265=&response-content-disposition=inline%3B+filename%3DReinforcement_learning_of_a_continuous_m.pdf&Expires=1631656637&Signature=XSIrhHHU8P-xhIxB3kY4iAfFGXd~fHCJUJEsDHz2V9xRd5VkBE~votOilglBrv2C7km5np4lZl71g90XMCVrDFGmuIq-R7cnYxbGdkWERA~FLorrKZ2iLET2otyf1oQ6tNT~G8BxS6~9qvIIemkIeeSqwQwi~LQwUxgvI181JOHCKQ~uhi129wHP-XVnDHRsxp2hJKUU1hb1XdshPU-bKvukTGSp7av86J2akQqq9s7oSboukMQT5gw9dgujyrST~LrcSTC1-AEmfJTJf1wvT5nefV~IhBbuH7WgHV1ITESCuODVgAKpi38f~fmUHTfLEqv7oCRH4N0XhoLaL8nnWw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

### Abstract
Abstract Reinforcement learning is the scheme for unsupervised learning in which robots are
expected to acquire behavior skills through self-explorations based on reward signals. There are
some difficulties, however, in applying conventional reinforcement learning algorithms to motion
control tasks of a robot because most algorithms are concerned with discrete state space and based
on the assumption of complete observability of the state. Real-world environments often have partial
observablility; therefore, robots have to estimate the unobservable hidden states. This paper proposes a
method to solve these two problems by combining the reinforcement learning algorithm and a learning
algorithm for a continuous time recurrent neural network (CTRNN). The CTRNN can learn spatio-
temporal structures in a continuous time and space domain, and can preserve the contextual flow
by a self-organizing appropriate internal memory structure. This enables the robot to deal with the
hidden state problem. We carried out an experiment on the pendulum swing-up task without rotational
speed information. As a result, this task is accomplished in several hundred trials using the proposed
algorithm. In addition, it is shown that the information about the rotational speed of the pendulum,
which is considered as a hidden state, is estimated and encoded on the activation of a context neuron.

### Notes
Slightly newer paper by the same people above
If looking at one, probably look at both

This methods works with/ accounts for hidden states

## A Systematic Literature Review of the Successors of NeuroEvolution of Augmenting Topologies

link: https://watermark.silverchair.com/evco_a_00282.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAuQwggLgBgkqhkiG9w0BBwagggLRMIICzQIBADCCAsYGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMpXGfL-hevp3gPdihAgEQgIIClyK9RUcBfZCKoIu3W6DdeYI6j4UQ_Bn5ulHrsUn1mG6GFXmB1DvklTwgShq8JNRkLzv7r5L-vMOEU2AFAPqF0fdS_PfqU9BPQGwJ3bv1xliGyBw8OD0rXSv1DmZAOqonDEqpcrF61zm5Y_r-jk4AU3W1HJRfeBQAkr1CTpirNURmkAW6gV178av2IHFXV8QzeQQ4oXt9qgcNOS4mAD6Wurw3Ni3wDhFtnHGcbnhzURLuo3Kf9XOniDBuyqYMgWeOQIrrxr6_3smomJwQJx6DmOOp8cRDkKziTae8nJB_VKcyIHaC47uycvaXUDeguFH4wbLT9kYaa0r6oRmuplfWPY-8y5PdX-wGpjSlE515yw5YETyMkKr_4_mk2Z1P6ToV6WK6vhhx1RhFFPXpjowDYRbGhl2JmkbCbY6xC57i7LadMWqZb3E_a7pJ-hMKClR2lQpSjl53F3zFYvODWbHyAeFESHugCBMkYr-hsWqBcM969H1DOFew4FTIpu4yPHQXPWv8DwjxutBmpoy0OL37GahkmrTTzdVmmkXbZzDP_VSt1wBF9XWhtxqaRXWW8HTPe6q8XlIGEEPI5-bgRPJX_zc3Te-lDGp0Jc7lhrUmk5oNn8vzaEOJHpO6thzFmZY-p95dewx8-QahokAWtz4BeOD7oznkHImqQ0iyXcMYo5K5-TJ9lEw0BfzgP9BgL1317HBFwYnzPs-Sy4872VCdIkjHYnEiEQrK2LImsEQSI1OwZxLmckBIvQm3SS9RTpIIto_Svx8CqPG8Z4z2Kxd9EGu9elUDRkR4z2Ewl23UTk5fh_hOvfxOSHOf9FIG2ZweQkPrVbH0NhKX4nQi_mXqXvOBMvXwMyq4VHyJRfaFx4Eft8MDJy64nA

### Abstract
NeuroEvolution (NE) refers to a family of methods for optimizing Artificial Neural Net-
works (ANNs) using Evolutionary Computation (EC) algorithms. NeuroEvolution of
Augmenting Topologies (NEAT) is considered one of the most influential algorithms in
the field. Eighteen years after its invention, a plethora of methods have been proposed
that extend NEAT in different aspects. In this article, we present a systematic literature
review (SLR) to list and categorize the methods succeeding NEAT. Our review protocol
identified 232 papers by merging the findings of two major electronic databases. Ap-
plying criteria that determine the paper’s relevance and assess its quality, resulted in
61 methods that are presented in this article. Our review article proposes a new cate-
gorization scheme of NEAT’s successors into three clusters. NEAT-based methods are
categorized based on 1) whether they consider issues specific to the search space or the
fitness landscape, 2) whether they combine principles from NE and another domain, or
3) the particular properties of the evolved ANNs. The clustering supports researchers
1) understanding the current state of the art that will enable them, 2) exploring new
research directions or 3) benchmarking their proposed method to the state of the art,
if they are interested in comparing, and 4) positioning themselves in the domain or
5) selecting a method that is most appropriate for their problem.

### Notes
May not be the most relavent, but I think Dr. Yoder will enjoy it!

NEAT-CTRNN Could be a point of comparison

## A Database for Learning Numbers by Visual Finger Recognition in Developmental Neuro-Robotics

link: https://www.frontiersin.org/articles/10.3389/fnbot.2021.619504/full#B27

### Abstract

     

Numerical cognition is a fundamental component of human intelligence that has not been fully understood yet. Indeed, it is a subject of research in many disciplines, e.g., neuroscience, education, cognitive and developmental psychology, philosophy of mathematics, linguistics. In Artificial Intelligence, aspects of numerical cognition have been modelled through neural networks to replicate and analytically study children behaviours. However, artificial models need to incorporate realistic sensory-motor information from the body to fully mimic the children's learning behaviours, e.g., the use of fingers to learn and manipulate numbers. To this end, this article presents a database of images, focused on number representation with fingers using both human and robot hands, which can constitute the base for building new realistic models of numerical cognition in humanoid robots, enabling a grounded learning approach in developmental autonomous agents. The article provides a benchmark analysis of the datasets in the database that are used to train, validate, and test five state-of-the art deep neural networks, which are compared for classification accuracy together with an analysis of the computational requirements of each network. The discussion highlights the trade-off between speed and precision in the detection, which is required for realistic applications in robotics.

### Notes
Use fingers and an embodied approach to improve number recognition

## A Developmental Neuro-Robotics Approach for Boosting the Recognition of Handwritten Digits

link: https://ieeexplore.ieee.org/document/9206857

## Abstract
Developmental psychology and neuroimaging research identified a close link between numbers and fingers, which can boost the initial number knowledge in children. Recent evidence shows that a simulation of the children's embodied strategies can improve the machine intelligence too. This article explores the application of embodied strategies to convolutional neural network models in the context of developmental neurorobotics, where the training information is likely to be gradually acquired while operating rather than being abundant and fully available as the classical machine learning scenarios. The experimental analyses show that the proprioceptive information from the robot fingers can improve network accuracy in the recognition of handwritten Arabic digits when training examples and epochs are few. This result is comparable to brain imaging and longitudinal studies with young children. In conclusion, these findings also support the relevance of the embodiment in the case of artificial agents' training and show a possible way for the humanization of the learning process, where the robotic body can express the internal processes of artificial intelligence making it more understandable for humans.

### Notes
Referenced in the above work uses embodied intelligence to improve learning results

Tries to mimic child learning

Gradually introduces more information

## Quadrupedal Locomotion: GasNets, CTRNNs and Hybrid CTRNN/PNNs Compared

link: https://d1wqtxts1xzle7.cloudfront.net/46480086/Quadrupedal_Locomotion_GasNets_CTRNNs_an20160614-31269-z3ihcl.pdf?1465920897=&response-content-disposition=inline%3B+filename%3DQuadrupedal_Locomotion_GasNets_CTRNNs_an.pdf&Expires=1631658256&Signature=d5ek95hpLPL8Sb07I55J-hw6p2ZDHKA1zilkfZvkACv7J1oPBUtZr355QQD6deGQWBMxpUTRAcThK4xybhlWOfqzJmye-Kmn~K~IUzRgWd13pJqZaC8NgnblqBtziMXYweP4mDLlFKZATm3p92h-CZuKyHZAuhkxYCGdacmTwQi1fdO6Z5JNhdJrt-6f0bUTeIXto0JZcBUxuqmExHRCQnRxPso~53Fflqvnw2gxXb2HZy22dxOS6AORjN0-f6-C~FuenbxAFDftVPGYHDtMmoNVxhbbpR6d80pZxn3LLBLI2NXBmHU6fBnuUwwEU0zi7UtbOW1C1gzq~F6CPNH~ZQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

### Abstract
Evolutionary Robotics seeks to use evolutionary techniques
to create both physical and physically simulated robots capa-
ble of exhibiting characteristics commonly associated with
living organisms. Typically, biologically inspired artificial
neural networks are evolved to act as sensorimotor control
systems. These networks include; GasNets, Continuous Time
Recurrent Neural Networks (CTRNNs) and Plastic Neural
Networks (PNNs). This paper seeks to compare the perfor-
mance of such networks in solving the problem of locomotion
in a physically simulated quadruped. The results in this paper,
taken together with those of other studies (summarized in this
paper) help us to assess the relative strengths and weaknesses
of the these three different approaches

###
Notes using CTRNNs in quadrupedal motion


