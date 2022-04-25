import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from pathlib import Path
import pdb


class Agent:
    def __init__(self):
        self.state = np.array([1, 1])
        # self.state = np.array([0, 0])

        # up, down, left, right
        self.action_space = {
            "up": 0,
            "down": 1,
            "left": 2,
            "right": 3
        }

    def update_state(self, next_state):
        self.state = next_state

    def avail_actions(self, state):
        avail_actions = dict(self.action_space)

        # if state along the top of the arena
        if state[0] == 0:
            del avail_actions["up"]

        # if state along the bottom of the arena
        if state[0] == 6:
            del avail_actions["down"]

        # if state along the right of the arena
        if state[1] == 6:
            del avail_actions["right"]

        # if state along the left of the arena
        if state[1] == 0:
            del avail_actions["left"]

        # return list of available actions
        action_list = []
        for action in avail_actions:
            action_list.append(avail_actions[action])

        return action_list

    def act(self):
        # take action based on observation
        avail_actions = self.avail_actions(self.state)
        probs = (1.0 / len(avail_actions)) * np.ones(len(avail_actions))
        action_idx = np.argmax(np.random.multinomial(1, probs))
        action = avail_actions[action_idx]

        return action


class POMDP:
    def __init__(self, params, seed=0):
        """
        Args:
            params (dict): all necessary parameters
            seed (int, optional): random seed. Defaults to 0.

        """
        self.params = params

        self.state_space = np.zeros((7, 7))
        self.goal_state = np.array([5, 2])
        self.t = 0
        self.t_max = 10000
        self.done = False
        self.agent = Agent()

    def state_transition(self, state, action):
        """transition function for POMDP
        """
        # choose the state the agent wants to go to
        if action == 0:
            chosen_next_state = state - np.array([1, 0])
        elif action == 1:
            chosen_next_state = state + np.array([1, 0])
        elif action == 2:
            chosen_next_state = state - np.array([0, 1])
        elif action == 3:
            chosen_next_state = state + np.array([0, 1])

        # states that are available to randomly transition to
        up_state = state - np.array([1, 0])
        down_state = state + np.array([1, 0])
        left_state = state - np.array([0, 1])
        right_state = state + np.array([0, 1])

        avail_states = [up_state, down_state, left_state, right_state]
        actual_avail_states = []
        for avail_state in avail_states:
            if not ((avail_state[0] < 0) or (avail_state[1] < 0) or (avail_state[1] > 6) or (avail_state[1] > 6) or (np.array_equal(chosen_next_state, avail_state))):
                actual_avail_states.append(avail_state)

        rand_value = np.random.rand()
        if rand_value <= 0.75:
            # transition to the chosen next state
            next_state = chosen_next_state
        else:
            # randomly transition to another available surrounding state
            probs = (0.25 / len(actual_avail_states)) * np.ones(len(actual_avail_states))
            state_idx = np.argmax(np.random.multinomial(1, probs))
            next_state = actual_avail_states[state_idx]

        return next_state

    def step(self, state, action):
        ep_reward = 0
        done = False

        next_state = self.state_transition(state, action)

        # update agent's real state
        self.agent.update_state(next_state)

        self.t += 1
        #TODO if agent is at the goal square or time runs out
        if np.array_equal(self.agent.state, self.goal_state) or (self.t >= self.t_max):
            done = True
            ep_reward = -1 * self.t

        # return environment rewards
        return ep_reward, done

    def reset(self):
        # reset everything for new episode
        self.t = 0
        self.agent.state = np.array([1, 1])
        self.done = False

    def set_seed(self):
        np.random.seed(self.seed)


def init_params():
    params = {
        "policy": "random_walk_policy",
        "n_episodes": 10000,
        "env": {
            "seed": 0,
        }
    }
    return params


def run_episode(env):
    # while not done, take action, get observation, etc.
    env.reset()
    done = False

    while not done:
        action = env.agent.act()
        ep_reward, done = env.step(env.agent.state, action)

    return ep_reward


def main():
    params = init_params()
    env = POMDP(params)

    ep_rewards = np.zeros(params["n_episodes"])
    for ep in range(params["n_episodes"]):
        ep_reward = run_episode(env)
        ep_rewards[ep] = ep_reward
        if ep % 500 == 0:
            print(f"finished episode {ep} / {params['n_episodes']} with reward {ep_reward}")

    # get statistics on the episode rewards
    print("saving")
    Path("output").mkdir(parents=True, exist_ok=True)
    np.save(f"output/ep_rewards_{params['policy']}", ep_rewards)


def reward_stats():
    Path("output").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)
    ep_rewards = np.load(f"output/ep_rewards_random_walk_policy.npy")

    mean = np.mean(ep_rewards)
    std = np.std(ep_rewards)
    max_val = np.max(ep_rewards)
    min_val = np.min(ep_rewards)

    label_str = f"random_walk_policy\n$\mu$ = {round(mean)}, $\sigma$ = {round(std)}\nmax = {round(max_val)}, min = {round(min_val)}"
    #TODO there's no way this is going to be a normal distribution
    sns.distplot(ep_rewards, kde=False, label=label_str)
    plt.legend()

    plt.xlabel("Episodic Reward")
    plt.ylabel("Counts")
    plt.savefig(f"figures/random_walk_policy_stats.png")

if __name__ == "__main__":
    # main()
    reward_stats()
