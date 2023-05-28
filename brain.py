import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import collections
import random


# --------------------------------------- #
# Replay Memory                           #
# --------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):
        # build a FIFO deque with maxlen=capacity
        self.buffer = collections.deque(maxlen=capacity)

    # add a transition tuple to the buffer
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # random sample batch_size items
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list
        # *transitions means unzip the tuples
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    # current size of the buffer
    def size(self):
        return len(self.buffer)


# -------------------------------------- #
# FCN, input: state, output: Q values    #
# -------------------------------------- #

class Net(nn.Module):
    # build the network with 1 hidden layer
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        # [b,n_states]-->[b,n_hidden]
        self.fc1 = nn.Linear(n_states, n_hidden)
        # [b,n_hidden]-->[b,n_actions]
        self.fc2 = nn.Linear(n_hidden, n_actions)

    # forward propagation
    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------------------- #
# Build the DQN agent                    #
# -------------------------------------- #

class DQN:
    # initialization
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon, epsilon_min, epsilon_decay,
                 target_update, device, path):
        # assign the parameters
        self.n_states = n_states  # number of features of the state
        self.n_hidden = n_hidden  # number of neurons in the hidden layer
        self.n_actions = n_actions  # number of actions

        self.learning_rate = learning_rate  # learning rate
        self.gamma = gamma  # discount factor, discount the reward of next state
        self.epsilon = epsilon  # epsilon-greedy policy, the probability of exploration
        self.epsilon_min = epsilon_min  # minimum epsilon
        self.epsilon_decay = epsilon_decay  # decay rate of epsilon

        self.target_update = target_update  # frequency of updating the target network
        self.device = device  # cpu or gpu
        self.count = 0  # count the number of updating the target network

        # build 2 network with the same structure but different parameters
        # instantiate the training network, output the Q value for each action
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        # instantiate the target network
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)

        # optimizer, update the parameters of the training network
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        # path to save the model
        self.path = path

    # take an action
    def take_action(self, state):
        # decay the epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # enrich the dimension with a new axis and convert to tensor
        state = torch.Tensor(state[np.newaxis, :])
        if np.random.random() > self.epsilon:
            # forward propagation, get the Q value for each action
            actions_value = self.q_net(state)
            # get the index of the action with the maximum Q value
            action = actions_value.argmax().item()
        # randomly choose an action if the random number is less than epsilon
        else:
            action = np.random.randint(self.n_actions)
        return action

    # train the network
    def update(self, transition_dict):  # sample from the replay buffer
        # get the current states, array_shape=[b,4]
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # get the actions taken at the current states, tuple_shape=[b], dimension expansion[b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        # get the rewards of the actions, tuple_shape=[b], dimension expansion[b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        # get the next states, array_shape=[b,4]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        # reach the terminal state or not, tuple_shape=[b], dimension expansion[b,1]
        terminates = torch.tensor(transition_dict['terminates'], dtype=torch.float).view(-1, 1)

        # get the Q values of the action taken at the current state[b,1]
        q_values = self.q_net(states).gather(1, actions)
        # get the maximum Q values of the next state[b], dimension expansion[b,1]
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # target Q values: r + gamma * max(q_values of next state)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - terminates)  # if terminates, q_targets = rewards

        # mse loss between the q_values and q_targets
        dqn_loss = torch.mean(f.mse_loss(q_values, q_targets))
        # gradient will accumulate, so clear the gradient
        self.optimizer.zero_grad()
        # back propagation
        dqn_loss.backward()
        # update the parameters
        self.optimizer.step()

        # update the target network
        if self.count % self.target_update == 0:
            # replace the parameters of the target network with the parameters of the training network
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())

        self.count += 1

    # save the model
    def save(self):
        torch.save(self.q_net.state_dict(), self.path)

    # load the model
    def load(self, model):
        self.q_net.load_state_dict(model)
        self.q_net.eval()
