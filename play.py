import gym
from brain import DQN
import torch

# GPU or CPU
device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")

# load the environment
env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]  # 4
n_actions = env.action_space.n  # 2

episode = 20

# agent parameters
n_hidden = 256  # number of neurons in the hidden layer
lr = 0  # learning rate
gamma = 0  # discount factor
epsilon = 0  # greedy factor, 0 means no exploration
epsilon_min = 0  # minimum epsilon
epsilon_decay = 0  # decay rate of epsilon
target_update = 0xFFFFFFFF  # update frequency of the target network, no need to update
path = 'models/trained_model.pth'  # path to the trained model

agent = DQN(n_states=n_states,
            n_hidden=n_hidden,
            n_actions=n_actions,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            target_update=target_update,
            device=device,
            path=path
            )
# load the trained model
model = torch.load(path)
agent.load(model)

for i in range(episode):
    # reset the environment
    state = env.reset()[0]
    # reset the reward
    episode_return = 0

    while True:
        # get the action based on the current state
        action = agent.take_action(state)
        # update the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        # update the current state
        state = next_state
        # update the reward
        episode_return += reward

        # end the episode if terminated or truncated
        if terminated or truncated:
            break

    # record the reward of each episode
    print("Episode", i+1, "reward:", episode_return)
