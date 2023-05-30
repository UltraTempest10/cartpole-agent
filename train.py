import gym
from brain import DQN, ReplayBuffer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


# GPU or CPU
device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")

# ------------------------------- #
# Hyperparameters                 #
# ------------------------------- #

iteration = 500  # number of episodes
capacity = 1000  # size of the replay buffer
batch_size = 128  # sample size from the replay buffer
min_size = 200  # minimum size to start training

n_hidden = 256  # number of neurons in the hidden layer
lr = 2e-4  # learning rate
gamma = 0.99  # discount factor
epsilon = 1.0  # greedy factor
epsilon_min = 0.01  # minimum epsilon
epsilon_decay = 0.999  # decay rate of epsilon
target_update = 200  # update frequency of the target network
path = 'model/your_model.pth'

return_list = []  # record the return of each episode


# load the environment
env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]  # number of features of the state, 4
n_actions = env.action_space.n  # number of actions, 2

# initialize the replay buffer
replay_buffer = ReplayBuffer(capacity)
# initialize the agent
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

# progress bar
with tqdm(total=iteration, desc='Training') as pbar:

    # training loop
    for i in range(iteration):
        # reset the environment
        state = env.reset()[0]
        # reset the reward
        episode_return = 0

        while True:
            # get the action based on the current state
            action = agent.take_action(state)
            # update the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            # add the transition to the replay buffer
            replay_buffer.add(state, action, reward, next_state, terminated)
            # update the current state
            state = next_state
            # update the reward
            episode_return += reward

            # training
            if replay_buffer.size() > min_size:
                # random sample from the replay buffer
                s, a, r, ns, t = replay_buffer.sample(batch_size)
                # build the transition dictionary
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'terminates': t,
                }
                # update the agent
                agent.update(transition_dict)

            # end the episode if terminated or truncated
            if terminated or truncated:
                break

        # record the return of each episode
        return_list.append(episode_return)

        # update the progress bar
        pbar.set_postfix({
            'reward': '%.3f' % return_list[-1],
        })
        pbar.update(1)

# save the trained model
agent.save()

# draw the return curve
episodes_list = list(range(len(return_list)))
plt.scatter(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN Returns')
plt.show()
