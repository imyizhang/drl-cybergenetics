# drl-cybergenetics

drl-cybergenetics is a research framework for the use Deep Reinforcement Learning (DRL) in the context of Cybergenetics, including for the control of Chemical Reaction Networks (CRNs) and co-cultures.

Among others, it includes:
- CRN and co-cultures custom simulation environments (and soon [Gym](https://github.com/openai/gym)-like and [dm_control](https://github.com/deepmind/dm_control)-like environments, compatible with high-performance DRL libraries like [Baselines](https://github.com/openai/baselines) and [ACME](https://github.com/deepmind/acme))
- [TensorFlow](https://github.com/tensorflow/tensorflow)-based custom implementations of state-of-the-art DRL agents, like DDQN and DDPG 

The custom environments and DRL agents are specified by independent configuration files.
