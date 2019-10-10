from Rainbow.agent import Agent
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

agent = Agent(
    "CartPole-v0",
    V_min=-10.0,
    V_max=10.0,
    batch_size=32,
    copy_every=100,
    n_step=3,
    lr=1e-3,
    num_atoms=51
)
results, losses = agent.train(5000)

fig, ax = plt.subplots(2, 1)
sns.lineplot(x=range(len(results)), y=results, ax=ax[0])
sns.lineplot(x=range(len(losses)), y=losses, ax=ax[1])
plt.show()
agent.show()
