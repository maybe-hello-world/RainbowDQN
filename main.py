from Rainbow.agent import Agent
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

agent = Agent("CartPole-v1", batch_size=30, copy_every=10)
results = agent.train(900)
sns.lineplot(x=range(len(results)), y=results)
plt.show()
agent.show()
