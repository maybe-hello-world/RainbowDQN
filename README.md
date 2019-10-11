# Rainbow DQN
This is Rainbow Deep Q-Learning implementation with PyTorch

Implemented articles:
* [x] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [x] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [x] [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
* [x] [Multi-step learning](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)
* [x] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [x] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
* [x] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* [x] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)

Articles are listed in the order in which commits are presented, every commit present one improvement to the algorithm.
There are two exceptions to that rule:
1. Commit with message "Rainbow" implements Distributional-RL, but as it was the last commit to combine algorithms it
named after the overall result.
2. Last commits implements some tweaks, performance hacks and description files adding (like readme, requirements, license etc.  

Feel free to open issues if (or more likely *when*) you find any problems or wrong implementation.  

## Acknowledgements
Special thanks to [@Kaixhin](https://github.com/Kaixhin) for C51 clear implementation and comments, on which version my one is based.