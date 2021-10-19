# Papers Viewed

Search Parameters: "Developmental neural networks", since 2021

## Towards self-organized control: Using neural cellular automata to robustly control a cart-pole agent

link: https://arxiv.org/pdf/2106.15240.pdf

Might be worth looking into

### Abstract

Neural cellular automata (Neural CA) are a recent framework used to model biological phenomena emerging from multicellular organisms. In these systems, artificial neural networks are used as update rules for cellular automata. Neural CA are end-to-end differentiable systems where the parameters of the neural network can be learned to achieve a particular task. In this work, we used neural CA to control a cart-pole agent. The observations of the environment are transmitted in input cells, while the values of output cells are used as a readout of the system. We trained the model using deep-Q learning, where the states of the output cells were used as the Q-value estimates to be optimized. We found that the computing abilities of the cellular automata were maintained over several hundreds of thousands of iterations, producing an emergent stable behavior in the environment it controls for thousands of steps. Moreover, the system demonstrated life-like phenomena such as a developmental phase, regeneration after damage, stability despite a noisy environment, and robustness to unseen disruption such as input deletion.

### Notes

This could be and interesting target for the learned CTRNNs
Use them to update a CA

The CA controls the pole agent

Claims that it a demonstrated developmental phase, regeneration after damage, stability despite a noisy environment, and robustness to unseen disruption such as input deletion. 

## Generating reward structures on a parameterized distribution of dynamics tasks

link: https://watermark.silverchair.com/isal_a_00466.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAu4wggLqBgkqhkiG9w0BBwagggLbMIIC1wIBADCCAtAGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMZGf_uw_qmS0_lJ7xAgEQgIICodlTuZaXFEB3gePBdJZucM8cJiMq7nFNt5xULHCFUs5WQvpIL2Ko51ljRMWHPAYvWZg5j9Okb-lE0D2IStKG1NZaFYg4HzjhgTM4dETrB5Hm-hG6tlnCrB5Y2xelGTyVaHNu7QfcutBiLofM6Rzd2HvDEjSGjJFjVbAd9Dn5qFQRAj9k1yF4XPl5eZPe-E6xXunpOLYmzUxEZ6GqsfvXy1unzcCiAQYRFLaWgnCEbZLmCYcCErRFzpysEvY9FcunyPOnJmSdPN4bO2uV7oSJwHOWA1ZUwzeJ7D3S40kGyyJakvPOhce9fmSCK57n9SCeFxf56P19FEC-cMP5ahdZHJN17jS2JrocQj65FPZ6jUv-vEuRiJeMz5JRiXZ5J3dwdiErikErkSYt6YynzPvQBUBcvB_pSqwTkGgSG973s1kTpd8ksORSZYxGvQScV9fLf90gqbHg3kJcyTwP5Xfb4m8uwZLED6VdcU-hycoLZkhmCntOrwsDDFHASagk3GE3Pj35QyatLtyPv0KZzqOm5kvHL5OesYS_ceqjCK1xLPd9JMgecA5I7LkLmpfJ-c_kkxDmbtB7yi5S4qUGeb0NU8JM7um7_1MgdxlPhSWU16TEcRcHaiRvtoa2RfGwbu-cRdc5yj9hV2TjIOhjdX4_kl-42SggrejB9a0nSnh7ofmXigOgSFWg_iY_rK1Tr7BTNhCzdK03w1GiVZvelIvm23GTJwrolym8Dyqd-KXshVymXRg8UCl4cV4uYISU0zBog_ayBreeWysZSWOU3kabSF9_K4dTE-blQzhm80RxdDyWckkVAmb9kXr8m7iKidWtShawYKl3oveqN78ah_Wz5cl4I3dgVSW0h3avDDAlSUlerzUo-NHbBKKkMm3iKoyETdU

### Abstract
In order to make lifelike, versatile learning adaptive in the
artificial domain, one needs a very diverse set of behaviors
to learn. We propose a parameterized distribution of classic
control-style tasks with minimal information shared between
tasks. We discuss what makes a task trivial and offer a basic
metric, time in convergence, that measures triviality. We then
investigate analytic and empirical approaches to generating
reward structures for tasks based on their dynamics in order to
minimize triviality. Contrary to our expectations, populations
evolved on reward structures that incentivized the most stable
locations in state space spend the least time in convergence
as we have defined it, because of the outsized importance our
metric assigns to behavior fine-tuning in these contexts. This
work paves the way towards an understanding of which task
distributions enable the development of learning

### Notes
Pretty abstract in concept, might be helpful in our reward structures




