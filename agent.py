"""
Policy Gradient Agent
"""

import numpy as np
import torch
from policy_network import PolicyNetwork_FC


class REINFORCEAgent(object):
    def __init__(self, gamma, lr, input_dims, n_actions, batch_size,
                 checkpoint_dir='tmp/dqn', algo=None, env_name=None):
        """
        Init the Agent parameter
        With decaying epsilon-greedy policy
        off-policy training
        Target policy: greedy
        behavioral policy: epsilon greedy

        :param gamma: discount factor
        :param lr: learning rate
        :param n_actions: number of action
        :param input_dims: input dimension
        :param batch_size: batch size for optimisation
        :param algo: algo name for storing the parameters
        :param env_name: gym env name
        :param checkpoint_dir: dir to save the weight
        """
        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.algo = algo
        self.env_name = env_name
        self.checkpoint = checkpoint_dir

        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        # Store batches of trajectory for gradient ascent
        self.total_rewards = []
        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []
        self.batch_counter = 1

        # current Q network
        self.policyNetwork = PolicyNetwork_FC(self.lr, self.n_actions,
                                              input_dims=self.input_dims,
                                              name=self.env_name + '_' + self.algo,
                                              cheakpoint_dir=self.checkpoint)

    def choose_action(self, observation):
        """
        choose action base on the current observation
        assume discrete action. can also applied to continuous action which define by the output pdf

        :param observation: observation from the env
        :return: the action to take base on epsilon-greedy policy
        """
        # [observation] to add one more dimension
        state = torch.tensor([observation], dtype=torch.float).to(self.policyNetwork.device)
        with torch.no_grad():
            # get the probability of the actions
            # i.e.
            # [0.24, 0.76]
            # Probability of 0.24 to take action 0 and Probability of 0.76 to take action 1
            action_probs = self.policyNetwork.forward(state).detach().numpy()
        # sample the action from the distribution
        action = np.random.choice(self.action_space, p=action_probs)

        return action

    def discounted_rewards(self, rewards):
        """
        Turn the immediate rewards obtain from the sequence into discounted rewards

        :param rewards: observation from the env
        :return: the discounted reward - mean of the sequence (baseline)
        """
        discounted_rewards = np.zeros_like(rewards)
        running_add = rewards[len(rewards)]
        # Reversed in time to get the cumulative discounted reward for each time step
        # G_t = γ * G_t+1 + R_t
        for t in reversed(range(0, len(rewards))):
            running_add = self.gamma * running_add + rewards[t]
            discounted_rewards[t] = running_add

        return discounted_rewards - discounted_rewards.mean()

    def store_trajectory(self, states, actions, rewards):
        """
        Store the transition (s,a,r,s') in memory

        :param states: observation
        :param actions: action
        :param rewards: reward
        :return: None
        """
        self.batch_states.append(states)
        self.batch_rewards.append(self.discounted_rewards(rewards))
        self.batch_actions.append(actions)
        self.batch_counter += 1

    def save_models(self):
        self.policyNetwork.save_checkpoint()

    def load_models(self):
        self.policyNetwork.load_checkpoint()

    def learn(self):
        """
        Method to learn from the batches
        Doesn't have to be bathces, but it would increase the stability
        TODO: Try no batches and update every episode
        """
        # if there's not enough batches, do nothing
        # always do in batch
        if self.batch_counter < self.batch_size:
            return

        # zero out gradient
        self.policyNetwork.optimizer.zero_grad()

        # batches of experience
        # terminals: terminal state flag
        batches_states_tensor = torch.tensor(self.batch_states).to(self.policyNetwork.device)
        batches_actions_tensor = torch.tensor(self.batch_actions).to(self.policyNetwork.device)
        batches_rewards_tensor = torch.tensor(self.batch_rewards).to(self.policyNetwork.device)

        # Calculate the gradient of fitness function by sampling from the gradient of log policy
        # grad(J(θ)) = E[G_t grad(ln π(a|s))]
        logprob = torch.log(self.policyNetwork.forward(batches_states_tensor))
        product = batches_rewards_tensor * logprob[np.arange(len(batches_actions_tensor)), batches_actions_tensor]

        # mean for expectation
        # -1 for gradient ascent
        loss = -1 * product.mean()

        # compute the gradient
        loss.backward()

        # update the parameter
        self.policyNetwork.optimizer.step()
        self.learn_step_counter += 1

        # reset the batches
        self.total_rewards = []
        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []
        self.batch_counter = 1
