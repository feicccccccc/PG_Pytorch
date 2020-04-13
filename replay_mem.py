"""
Reference and credit:
https://www.youtube.com/watch?v=UlJzzLYgYoE

A memory class to store all the (s,a,r,s') tuple
Similar to a TD(0) approach, batch of experience is being record and randomized to reduce correlation between steps
Used in the optimisation

Mainly constructed by the Agent class
"""

import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        """
        Init the class
        :param max_size: Maximum number of experience
        :param input_shape: input shape
        :param n_actions: number of action
        """
        self.mem_size = max_size
        self.mem_counter = 0

        # (s,a,r,s') experience
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Method to save the experience
        :param state: observation
        :param action: action
        :param reward: immediate reward
        :param next_state: next observation
        :param done: identify terminal state or not
        :return: None
        """
        index = self.mem_counter % self.mem_size  # circular buffer
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        """
        Get a random batch of experience for optimization
        :param batch_size: batch size
        :return: experience in batch
        """
        max_mem = min(self.mem_counter, self.mem_size)
        # Generate random idx from the memory
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminals
