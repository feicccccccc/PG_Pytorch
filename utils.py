"""
Reference and credit:
https://www.youtube.com/watch?v=UlJzzLYgYoE

Utilities to wrap env with preprocess of the observation
Ref: https://github.com/openai/gym/tree/master/gym/wrappers
And also plot the learning curve
"""

import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0,
                 fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs

        return obs


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        """
        Base on what is the desired output
        i.e. CartPole output 4 number, the render is a (800,1200,3) on my computer and varies across different OS
        :param observation: original observation
        :return: processed Image
        """
        new_frame = self.render(mode='rgb_array')
        # convert to grey scale / eliminate the color channel dim
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        # downscale the image to desired output
        # Check the dimension
        resized_screen = cv2.resize(new_frame, (self.shape[2], self.shape[1]), interpolation=cv2.INTER_AREA)
        # change it back to (channel, height, width) to fit Pytorch convention
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        # normalisation
        new_obs = new_obs / 255.0

        return new_obs


class StackFrames(gym.ObservationWrapper):
    """
    Create a buffer to store previous frame
    """
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.repeat = repeat
        self.img_stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.img_stack.clear()
        observation = self.env.reset()
        # repeat the start frame in the image stack
        for _ in range(self.img_stack.maxlen):
            self.img_stack.append(observation)
        _, dimh, dimw = observation.shape  # reduce the first dimension
        return np.array(self.img_stack).reshape((self.repeat, dimh, dimw))

    def observation(self, observation):
        self.img_stack.append(observation)
        _, dimh, dimw = observation.shape  # reduce the first dimension
        return np.array(self.img_stack).reshape((self.repeat, dimh, dimw))


def make_env(env_name, shape=(200, 300, 1), repeat=4, clip_rewards=False, no_ops=0, fire_first=False):
    """
    Create the env with different wrapper for state preprocessing
    :param env_name: Gym env id
    :param shape: desired output shape
    :param repeat: number of stack frame
    :param clip_rewards: clip reward flag
    :param no_ops:
    :param fire_first:
    :return: env with wrapper
    """
    env = gym.make(env_name)
    # env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)  # Turn to grey and compress
    env = StackFrames(env, repeat)  # stack frame
    return env
