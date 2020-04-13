"""
REINFORCE with baseline using Pytorch

Reference and credit:
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

General Idea:
Find the derivative w.r.t. fitness/reward function, and use gradient ascent to find the policy that optimise the reward
The gradient can be estimated from the experience using policy gradient theorem

Prove of policy gradient theorem is tricky and not trivial since the Value function itself depends on the policy
Many reference ignore it except this beautiful explanation
https://lilianweng.github .io/lil-log/2018/04/08/policy-gradient-algorithms.html

grad(J(θ)) = E[Q(s,a) grad(ln π(a|s))]

"""

import gym
import numpy as np
from agent import REINFORCEAgent
from utils import plot_learning_curve, make_env

NUMBER_OF_FRAME = 4

if __name__ == '__main__':

    # env = make_env("CartPole-v0")
    env = gym.make("CartPole-v1")  # same env with different registry, terminate reward is larger (500)

    init_screen = env.reset()
    best_score = -np.inf
    load_checkpoint = False  # if user want to restart from checkpoint
    learn = True
    n_games = 1000  # number of episode

    agent = REINFORCEAgent(gamma=0.99, lr=0.0001,
                           input_dims=env.observation_space.shape,
                           n_actions=env.action_space.n, batch_size=8,
                           checkpoint_dir='models/', algo='REINFORCEAgent',
                           env_name='CartPole-v1-FC')

    if load_checkpoint:
        agent.load_models()

    # For record
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        observation = env.reset()
        states = []  # to store all the states in the current episode / trajectory
        rewards = []  # to store all the rewards in the current episode / trajectory
        actions = []  # to store all the actions in the current episode / trajectory
        done = False

        score = 0
        while not done:
            env.render()
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward

            # Record the time step in the trajectory
            states.append(observation)
            rewards.append(reward)
            actions.append(action)

            observation = next_observation
            n_steps += 1

        # Monte carlo update at the end of the episodes
        if learn:
            # store the trajectory
            agent.store_trajectory(observation, actions, rewards)
            # learn through the experience (if there's enough batches)
            agent.learn()

        # After 1 episode finish
        # keep record stuff
        scores.append(score)
        steps_array.append(n_steps)

        # Average score from last 100 episode
        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score, 'steps', n_steps)

        if avg_score > best_score:
            if learn:
                agent.save_models()
            best_score = avg_score

        if load_checkpoint and not learn and n_steps >= 18000:
            break

    # for graph piloting
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    # plot_learning_curve(steps_array, scores, figure_file)
