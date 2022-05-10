import argparse
import numpy as np
import scipy.stats as stats
from numpy.random import default_rng
import os
import pdb

# run all the parts in parallel, save run data to disk, then make a separate script for plotting (not in this script, make a separate file!)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--policy", type=int, default=1, help="Specify problem part to run")
parser.add_argument("-v", "--verbose", type=bool, default=True, help="Prints outputs if true")
args = parser.parse_args()


class Environment:
    def __init__(self):
        self.action_space_size = 3
        self.max_t = 1000

    def reset(self):
        self.env_t = 0
        done = False
        reward = 0

        return reward, done

    def step(self, action):
        done = 0
        if action == 0:
            reward = 0.95

        elif action == 1:
            # draw reward from normal distribution, mean=1, variance=1
            mean = 1
            std_dev = 1
            variance = std_dev ** 2
            reward = default_rng().normal(loc=mean, scale=variance)

        elif action == 2:
            # draw from distribution s.t. r = 0 w.p. 99% and r = 101 w.p. 1%
            rand_val = default_rng().uniform(0.0, 1.0)
            if rand_val >= 0.99:
                reward = 101
            else:
                reward = 0

        if self.env_t >= self.max_t:
            done = 1

        return reward, done


class Agent:
    def __init__(self, policy, action_space_size):
        self.action_space_size = action_space_size
        self.policy = policy

        self.action_to_idx = {
            "arm_1": 0,
            "arm_2": 1,
            "arm_3": 2
        }

        self.__init_learning(self.policy)

    def __init_learning(self, policy):
        self.expected_reward = np.zeros(self.action_space_size)
        self.n_samples_per_action = np.zeros(self.action_space_size)

        if self.policy == 1:
            self.exploration_samples_per_action = 5
            self.commit = False

        elif self.policy == 2:
            self.exploration_samples_per_action = 100
            self.commit = False

        elif self.policy == 3:
            self.epsilon = 0.01

        elif self.policy == 4:
            self.epsilon = 0.5

        elif self.policy == 5:
            self.epsilon = 0.5

        elif self.policy == 6:
            self.all_rewards = {
                0: [],
                1: [],
                2: []
            }

        elif self.policy == 7:
            pass

    def act(self, env_t):
        # take action WRT current policy
        exploration_bonus = 0
        if (self.policy == 1) or (self.policy == 2):
            # explore then commit policy
            if np.any(self.n_samples_per_action < self.exploration_samples_per_action):
                for sampled_action in range(self.action_space_size):
                    if self.n_samples_per_action[sampled_action] < self.exploration_samples_per_action:
                        # pick the action and stop the loop to actually take the action
                        action = sampled_action
                        break
            else:
                # take best action
                self.commit = True
                action = np.argmax(self.expected_reward)

        elif (self.policy == 3) or (self.policy == 4) or (self.policy == 5):
            # epsilon greedy action
            if self.policy == 5:
                # over time (for policy 5 at least), epsilon goes to 0 so we're eventually always taking the best action
                self.epsilon = 0.5 / (env_t + 1)

            rand_val = default_rng().uniform(0.0, 1.0)
            if rand_val <= self.epsilon:
                # take random action from uniform distribution over actions with probability epsilon
                action = np.random.randint(0, self.action_space_size)
            else:
                # take best action with probability 1-epsilon
                action = np.argmax(self.expected_reward)

        elif (self.policy == 6):
            upper_confidence_bounds = self._compute_upper_confidence_bounds(self.all_rewards)
            action = np.argmax(upper_confidence_bounds)
            exploration_bonus = self._exploration_bonus(action)

        elif (self.policy == 7):
            action = 1

        self.n_samples_per_action[action] += 1

        return action, exploration_bonus

    def online_learning(self, action, augmented_reward):
        # augmented_reward = reward + exploration_bonus
        if self.policy != 6:
            # compute expected augmented reward
            prev_reward = self.expected_reward[action]
            self.expected_reward[action] = (prev_reward + augmented_reward) / self.n_samples_per_action[action]

        elif self.policy == 6:
            # record all augmented rewards for later analysis
            self.all_rewards[action].append(augmented_reward)

    def _compute_upper_confidence_bounds(self, all_rewards):
        """
        Args:
            all_rewards (dict): dictionary of lists of augmented rewards received for each action up to current time
        """
        # assume all data drawn from normal distribution
        ## more-correct is to a t-distribution when below a certain sample size N, then switch to a normal distribution when we have more than N samples due to the central limit theorem
        upper_confidence_bounds = np.zeros(self.action_space_size)

        for action in range(self.action_space_size):
            rewards = all_rewards[action]
            # at the beginning we don't have any reward samples, so assume infinite upper confidence bound
            if len(rewards) == 0:
                upper_confidence_bound = np.inf
            else:
                # there is a 95% chance the true expected reward for this action falls in this range
                confidence_level = 0.95
                dof = len(rewards) - 1
                sample_mean = np.mean(rewards)
                sample_standard_error = stats.sem(rewards)
                confidence_interval = stats.norm.interval(alpha=confidence_level, loc=sample_mean, scale=sample_standard_error)
                upper_confidence_bound = confidence_interval[1]

            upper_confidence_bounds[action] = upper_confidence_bound

        return upper_confidence_bounds

    def _exploration_bonus(self, action):
        bonus = 0.5 / (self.n_samples_per_action[action] + 1)
        return bonus


def run_episode(env, agent):
    # summed reward up to the current time step
    ## sum_{t=0}^{T_curr-1} R(a_t)
    collected_reward = 0
    collected_reward_out = np.zeros(env.max_t)
    _, done = env.reset()

    while not done:
        action, exploration_bonus = agent.act(env.env_t)
        reward, done = env.step(action)
        augmented_reward = reward + exploration_bonus

        # only do online learning in certain cases for policies 1 and 2
        if (agent.policy == 1) or (agent.policy == 2):
            if agent.commit == False:
                agent.online_learning(action, augmented_reward)
        else:
            # always do online learning for other policies
            agent.online_learning(action, augmented_reward)

        # we learn from the augmented reward, but only report the summed environment-given rewards
        ## exploration_bonus can be considered an "intrinsic reward"
        if env.env_t >= env.max_t:
            done = True
            break

        collected_reward += reward
        collected_reward_out[env.env_t] = collected_reward

        env.env_t += 1

    return collected_reward_out

#TODO ucb is super slow compared to other policies, maybe see if there are ways to speed it up?
## basically any draw from random numbers is super slow. You could pre-generate these in an array, then just use those values, but its just a homework.

def main():
    # run a bunch of episodes
    env = Environment()
    n_episodes = 10000

    # record the rewards at each timestep
    episode_rewards = np.zeros((n_episodes, env.max_t))

    for episode in range(n_episodes):
        # init new agent for each episode so it doesn't remember anything from previous episodes
        agent = Agent(args.policy, env.action_space_size)
        episode_reward = run_episode(env, agent)

        # episode outputs
        episode_rewards[episode, :] = episode_reward
        if args.verbose:
            if episode % 10 == 0:
                print(f"Policy: {args.policy} --- Episode: {episode} / {n_episodes}")
            # print(f"Policy: {args.policy} --- Episode {episode} ended with {summed_episode_reward} reward")


    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    np.save(f"output/episode_reward_policy_{args.policy}", episode_rewards)



if __name__ == "__main__":
    main()
