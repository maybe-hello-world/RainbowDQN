from Rainbow.agent import Agent
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

agent = Agent("CartPole-v1", batch_size=50, copy_every=64)
results, losses = agent.train(8192)

fig, ax = plt.subplots(2, 1)
sns.lineplot(x=range(len(results)), y=results, ax=ax[0])
sns.lineplot(x=range(len(losses)), y=losses, ax=ax[1])
plt.show()
agent.show()
