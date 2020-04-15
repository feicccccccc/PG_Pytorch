import numpy as np
import torch


def discounted_rewards(rewards):
    """
    Turn the immediate rewards obtain from the sequence into discounted rewards

    :param rewards: observation from the env
    :return: the discounted reward - mean of the sequence (baseline)
    """
    discounted_rewards = np.zeros_like(rewards)
    running_add = rewards[len(rewards) - 1]
    discounted_rewards[[len(rewards) - 1]] = running_add
    # Reversed in time to get the cumulative discounted reward for each time step
    # G_t = γ * G_t+1 + R_t
    for t in reversed(range(0, len(rewards) - 1)):
        running_add = 0.9 * running_add + rewards[t]
        discounted_rewards[t] = running_add

    return discounted_rewards

if __name__ == '__main__':
    test = np.array([1.,1.,1.,1.,1.,1.,1.])
    print(discounted_rewards(test))