"""
Policy Gradient Agent
"""

import numpy as np
import torch
from policy_network import PolicyNetwork_FC


class REINFORCEAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-4,
                 replace=1000, algo=None, env_name=None, checkpoint_dir='tmp/dqn'):
        """
        Init the Agent parameter
        With decaying epsilon-greedy policy
        off-policy training
        Target policy: greedy
        behavioral policy: epsilon greedy

        :param gamma: discount factor
        :param epsilon: epsilon-greedy policy for the behavioural network
        :param lr: learning rate
        :param n_actions: number of action
        :param input_dims: input dimension
        :param mem_size: maximum number of experience store inside memory
        :param batch_size: batch size for optimisation
        :param eps_min: minimum epsilon
        :param eps_dec: epsilon decay rate
        :param replace: How many step to update the target policy network
        :param algo:
        :param env_name: gym env name
        :param checkpoint_dir: dir to save the weight
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_counter = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint = checkpoint_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        # current Q network
        self.q_eval = PolicyNetwork_FC(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name=self.env_name+'_'+self.algo+'_q_eval',
                                       cheakpoint_dir=self.checkpoint)
        # next Q network
        self.q_next = PolicyNetwork_FC(self.lr, self.n_actions,
                                       input_dims=self.input_dims,
                                       name=self.env_name+'_'+self.algo+'_q_next',
                                       cheakpoint_dir=self.checkpoint)

    def choose_action(self, observation, absolute_greedy=False):
        """
        choose action base on the current observation
        epsilon decay from high to eps_min
        absolute_greedy flag can turn behavioral policy to target policy, which in general maybe totally different
        :param absolute_greedy: flag to determine action. Turn behavioural network to target network
        :param observation: observation from the env
        :return: the action to take base on epsilon-greedy policy
        """
        if absolute_greedy:
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            with torch.no_grad():
                actions = self.q_eval.forward(state)
            action = actions.argmax().item()
            return action

        if np.random.random() > self.epsilon:
            # move the observation to the network's device
            # [] to add batch dimension
            state = torch.tensor([observation], dtype=torch.float).to(self.q_eval.device)
            # no need to use autograd since we are going to do it again in experience replay
            # not optimal but doable
            with torch.no_grad():
                actions = self.q_eval.forward(state)
            # get the indices of the maximum output (probability)
            # i.e.
            # a: tensor([0.7875, 0.9929])
            # a.argmax(): tensor(1)
            action = actions.argmax().item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store the transition (s,a,r,s') in memory
        :param state: observation
        :param action: action
        :param reward: reward
        :param next_state: next observation
        :param done: terminate state?
        :return: None
        """
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        """
        Sample experience for the experience replay class
        :return: batches of replay
        """
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer(self.batch_size)

        # change the transition to torch tensor for further processing
        states_tensor = torch.tensor(states).to(self.q_eval.device)
        rewards_tensor = torch.tensor(rewards).to(self.q_eval.device)
        # cast to torch.bool to remove warning
        # TODO: change the original numpy array dtype as well
        terminals_tensor = torch.tensor(terminals, dtype=torch.bool).to(self.q_eval.device)
        actions_tensor = torch.tensor(actions).to(self.q_eval.device)
        next_states_tensor = torch.tensor(next_states).to(self.q_eval.device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, terminals_tensor

    def replace_target_network(self):
        """
        Update the target network occasionally
        just copying the the behavioral network weight to the target network
        :return: None
        """
        if self.learn_step_counter % self.replace_target_counter == 0:
            # print("Replacing target network weight with eval network")
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        # hmmm interesting, linear decrement but not exponential decay.
        # TODO: try exponential decay?
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        # if there's not enough experience, do nothing
        # always do in batch
        if self.memory.mem_counter < self.batch_size:
            return

        # zero out gradient
        self.q_eval.optimizer.zero_grad()
        # Replace the target network occasionally
        self.replace_target_network()

        # batches of experience
        # terminals: terminal state flag
        states, actions, rewards, next_states, terminals = self.sample_memory()
        indices = np.arange(self.batch_size)

        # Estimate Q value
        # Get the network output actions value from experience
        q_pred = self.q_eval.forward(states)[indices, actions]
        # Bellmen equation
        # for (s,a,r,s') tuple
        # Q(s,a) = r + max(a')Q(s',a')
        # eval on q_next, to avoid updating the target network which may result in local optimal
        # This part is actually tricky.
        # If you are looking at the code, you may want to refer to the original paper for more information
        # why there are two network to eval the Q(s,a)
        q_next = self.q_next.forward(next_states).max(dim=1)[0]

        # Change the reward of terminal state to 0
        q_next[terminals] = 0.0
        # 1 step look ahead
        q_target = rewards + self.gamma * q_next

        # learning loss between the TD target and the current estimate
        loss = self.q_eval.loss(q_pred, q_target.detach())  # no autograd on q_target
        # compute the gradient
        loss.backward()

        # update the parameter
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
