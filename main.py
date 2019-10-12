from Rainbow.agent import Agent
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

agent = Agent(
    "CartPole-v0",
    V_min=-10.0,
    V_max=10.0,
    batch_size=32,
    copy_every=500,
    play_before_learn=1000,
    n_step=2,
    lr=1e-3,
    num_atoms=51,
    noisy=False
)
results, losses = agent.train(5000)

fig, ax = plt.subplots(2, 1)
sns.lineplot(x=range(len(results)), y=results, ax=ax[0])
sns.lineplot(x=range(len(losses)), y=losses, ax=ax[1])
ax[0].set_title('Results')
ax[1].set_title('Losses')
plt.show()
agent.show()
