"""
Reference and credit:
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

General Idea:
Find the derivative w.r.t. fitness/reward function, and use gradient ascent to find the policy that optimise the reward
The gradient can be estimated from the experience using policy gradient theorem

Prove of policy gradient theorem is tricky and not trivial since the Value function itself depends on the policy
Many reference ignore it except this beautiful explanation
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

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
    load_checkpoint = True  # if user want to restart from checkpoint
    greedy_action = False  # use behavioural policy / target policy
    learn = True
    initial_epsilon = 0.5
    n_games = 1000  # number of episode

    agent = REINFORCEAgent(gamma=0.99, epsilon=initial_epsilon, lr=0.0001,
                           input_dims=env.observation_space.shape,
                           n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                           batch_size=128, replace=200, eps_dec=5e-5,
                           checkpoint_dir='models/', algo='REINFORCEAgent',
                           env_name='CartPole-v0-FC')

    if load_checkpoint:
        agent.load_models()

    # For record
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            env.render()
            action = agent.choose_action(observation, greedy_action)
            next_observation, reward, done, info = env.step(action)
            score += reward

            if learn:
                # store the transition (s,a,r,s') inside the replay memory
                agent.store_transition(observation, action, reward, next_observation, done)
                # learn through the experience (if there's enough batches)
                agent.learn()
            observation = next_observation
            n_steps += 1

        # After 1 episode finish
        # keep record stuff
        scores.append(score)
        steps_array.append(n_steps)

        # Average score from last 100 episode
        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if learn:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if load_checkpoint and not learn and n_steps >= 18000:
            break

    # for graph piloting
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    plot_learning_curve(steps_array, scores, eps_history, figure_file)
