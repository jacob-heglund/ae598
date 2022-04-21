import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from pathlib import Path
import pdb


class Agent:
    def __init__(self, initial_state, initial_belief_state, policy, q_optimal, state_space, action_space):
        self.state = initial_state
        self.belief_state = initial_belief_state
        self.policy = policy
        self.q_optimal = q_optimal
        self.state_space = state_space
        self.action_space = action_space

    def update_state(self, next_state):
        self.state = next_state

    def _belief_state_transition(self, obs, action):
        # update beliefs of agent based on observations and past beliefs
        if action == 0:
            # switch belief state entries
            self.belief_state = np.flip(self.belief_state)
        elif action == 1:
            # probabilistic transition
            entry_1 = 0.9 * self.belief_state[0] + 0.1 * self.belief_state[1]
            entry_2 = 0.1 * self.belief_state[0] + 0.9 * self.belief_state[1]
            self.belief_state = np.array([entry_1, entry_2])
        elif action == 2:
            if self.state == 0:
                self.belief_state = np.array([1.0, 0.0])
            elif self.state == 1:
                self.belief_state = np.array([0.0, 1.0])

    def optimal_policy_full_obs(self, state):
        # I computed the optimal policy on paper since it is really easy
        ## We have a stationary policy since we're acting over infinite time horizon, so 1 unique action per state
        ## and we have only 2 states. The optimal policy is to take whatever action maximizes your immediate reward since there are no longer-term planning requirements.
        if state == 0:
            # if in state 0, stay where you are
            action = 1
        elif state == 1:
            # if in state 1, switch to the other state
            action = 0

        return action

    def optimal_q_full_obs(self):
        # do the averaging V(a) = \sum_s b_t(s) Q(s, a)
        belief = np.expand_dims(self.belief_state, 1)
        q = self.q_optimal.T
        v_a = q @ belief
        action = np.argmax(v_a)

        return action

    def act(self, obs):
        # take action based on observation
        if self.policy == "greedy_policy":
            ## policy 1: The policy acts according to the most likely belief state, pi(s_b)
            ### where s_b = argmax_{s \ in S} b_t(s)
            most_likely_state = np.argmax(self.belief_state)
            action = self.optimal_policy_full_obs(most_likely_state)

        elif self.policy == "stochastic_policy":
            ## policy 2: First draw a state as a categorical random variable with probabilities given by the belief state, then act according to the drawn state using the optimal policy, pi(s_b)
            ### where s_b \sim b_t(s) (random draw of state with probabilities given as the belief state probabilities)
            drawn_state = np.argmax(np.random.multinomial(1, self.belief_state))
            action = self.optimal_policy_full_obs(drawn_state)

        elif self.policy == "greedy_value":
            ## policy 3: need to first compute the optimal Q(s, a) for all state-action pairs assuming fully observable (but still stochastic transitions), then use that as part of the agent's actual policy
            ## we can define the expected action-value as V(a) = \sum_s b_t(s) Q(s, a), which is only a function of action
            ###  we then take the action which maximizes V(a) at each timestep
            action = self.optimal_q_full_obs()

        # update belief based on observation and selected action
        self._belief_state_transition(obs, action)
        return action


class POMDP:
    def __init__(self, params, gamma=0.99, t_max=1000, seed=0):
        """ 2-state POMDP
        Args:
            params (dict): all necessary parameters
            t_max (int, optional): max number of episode time steps. Defaults to 1000.
            gamma (float, optional): discount factor for rewards. Defaults to 0.99.
            seed (int, optional): random seed. Defaults to 0.

        """
        self.params = params

        self.state_space = [0, 1]
        self.action_space = [0, 1, 2]
        self.n_states = len(self.state_space)
        self.n_actions = len(self.action_space)

        self.STATE_TO_IDX = {
            "good": 0,
            "bad": 1
        }

        self.ACTION_TO_IDX = {
            "switch": 0,
            "stay": 1,
            "locate": 2
        }

        self.IDX_TO_STATE = dict((v, k) for k, v in self.STATE_TO_IDX.items())
        self.IDX_TO_ACTION = dict((v, k) for k, v in self.ACTION_TO_IDX.items())

        # probability that next state == current state given action == "stay"
        self.stay_prob = 0.9
        self.gamma = self.params["env"]["gamma"]
        self.rewards = []
        self.t = 0
        self.t_max = self.params["env"]["t_max"]
        self.done = False

        self._init_tabular_reward_function()
        self._init_tabular_transition_function()
        self._init_tabular_observation_function()

        # q-value iteration parameters
        self.n_iterations = 0
        self.err = 0.001
        self._value_iteration()

        self.agent = Agent(params["initial_state"], params["initial_belief_state"], params["policy"], self.q, self.state_space, self.action_space)

    def _init_tabular_reward_function(self):
        # initialize reward function as a matrix of size S x A
        # reward = 0
        # if state == 0:
        #     if action == 1:
        #         reward = 5
        #     elif action == 2:
        #         reward = -1
        # elif state == 1:
        #     if action == 1:
        #         reward = -5
        #     elif action == 2:
        #         reward = -1

        self.R = np.zeros((self.n_states, self.n_actions))
        self.R[0, 1] = 5
        self.R[0, 2] = -1
        self.R[1, 1] = -5
        self.R[1, 2] = -1

    def _init_tabular_transition_function(self):
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))

        # deterministic transition to other state with "switch" action
        self.P[0, 0, 1] = 1
        self.P[1, 0, 0] = 1

        # probabilistic transition with "stay"
        self.P[0, 1, 0] = self.stay_prob
        self.P[0, 1, 1] = 1.0 - self.stay_prob

        self.P[1, 1, 1] = self.stay_prob
        self.P[1, 1, 0] = 1.0 - self.stay_prob

        # deterministic transition with "locate"
        self.P[0, 2, 0] = 1
        self.P[1, 2, 1] = 1

    def _init_tabular_observation_function(self):
        # probability of observing obs_value[state, action]
        self.O = np.zeros((self.n_states, self.n_actions))
        # the value the agent will observe
        self.obs_values = np.zeros((self.n_states, self.n_actions))

        self.O[:, 0] = 1
        self.O[:, 1] = 1
        self.obs_values[:, 0] = -1
        self.obs_values[:, 1] = -1

        self.O[:, 2] = 1
        self.obs_values[0, 2] = 0
        self.obs_values[1, 2] = 1

    def _value_iteration(self):
        self.q = np.zeros((len(self.state_space), len(self.action_space)))
        self.curr_err = np.inf

        while self.curr_err >= self.err:
            q_prev = np.copy(self.q)

            for state in range(len(self.state_space)):
                for action in range(len(self.action_space)):
                    sum_term = 0
                    for next_state in range(len(self.state_space)):
                        reward = self.R[state, action]
                        max_term = self.gamma * np.max(self.q[next_state, :])
                        sum_term += self.P[state, action, next_state] * (reward + max_term)

                    self.q[state, action] = sum_term

            self.curr_err = np.max(self.q - q_prev)
            self.n_iterations += 1

    def state_transition(self, state, action):
        """transition function for POMDP

        Args:
            state (int): s \in {0, 1}, describes current state
            action (int): a \in {0, 1, 2}, describes current action

        Returns:
            int: s' \in {0, 1}, describes next state
        """
        if state == 0:
            if action == 0:
                # deterministic transition to other state with "switch"
                next_state = 1

            elif action == 1:
                # probabilistic transition with "stay"
                rand_value = np.random.rand()
                if rand_value <= 0.9:
                    next_state = state
                else:
                    next_state = 1

            elif action == 2:
                # deterministic transition with "locate"
                next_state = state

        if state == 1:
            if action == 0:
                # deterministic transition to other state with "switch"
                next_state = 0

            elif action == 1:
                # probabilistic transition with "stay"
                rand_value = np.random.rand()
                if rand_value <= 0.9:
                    next_state = state
                else:
                    next_state = 0

            elif action == 2:
                # deterministic transition with "locate"
                next_state = state

        return next_state

    def observation(self, state, action):
        """observation function for POMDP

        Args:
            state (int): s \in {0, 1}, describes current state
            action (int): a \in {0, 1, 2}, describes current action

        Returns:
            int: obs \in {0, 1, -1}, describes observation given previous state and action, -1 is an "error" observation which gives no information about the current state
        """
        return self.obs_values[state, action]

    def reward(self, state, action):
        # agent can't see the immediate reward to make decisions (this is an artificial constraint from Prof. Ornik's definition of the problem), only the discounted episode reward when the episode finishes
        return self.R[state, action]

    def episode_reward(self, rewards, gamma):
        # get discounted gammas for each timestep
        gammas = gamma * np.ones(self.t)
        powers = np.arange(0, self.t, step=1)
        discounted_gamma = np.power(gammas, powers)

        rewards = np.asarray(rewards)

        # summed discounted rewards
        ep_reward = np.dot(discounted_gamma, rewards)
        return ep_reward

    def step(self, state, action):
        ep_reward = 0
        done = False

        next_state = self.state_transition(state, action)
        # update agent's real state
        self.agent.update_state(next_state)
        # get observation at next_state
        obs = self.observation(state, action)
        self.rewards.append(self.reward(state, action))

        self.t += 1

        if self.t >= self.t_max:
            done = True
            ep_reward = self.episode_reward(self.rewards, self.gamma)

        # return environment observations
        return obs, ep_reward, done

    def reset(self):
        # reset everything for new episode, return initial observation
        self.t = 0
        self.rewards = []
        self.agent.state = self.params["initial_state"]
        self.agent.belief_state = self.params["initial_belief_state"]
        self.done = False

    def set_seed(self):
        np.random.seed(self.seed)

    def render(self):
        pass



def init_params():
    params = {
        "initial_state": 0,
        "initial_belief_state": np.array([0.50, 0.50]),
        "policy": "greedy_policy",
        # "policy": "stochastic_policy",
        # "policy": "greedy_value",
        "n_episodes": 1000,
        "env": {
            "t_max": 1000,
            "seed": 0,
            "gamma": 0.99
        }
    }
    return params


def run_episode(env):
    # reset to get initial observation

    # while not done, take action, get observation, etc.
    obs = env.reset()
    done = False

    while not done:
        action = env.agent.act(obs)
        obs, ep_reward, done = env.step(env.agent.state, action)

    return ep_reward


def main():
    params = init_params()
    policies = ["greedy_policy", "stochastic_policy", "greedy_value"]
    initial_belief_state_0 = np.arange(0.05, 1.05, 0.1)

    for policy in policies:
        ep_rewards = np.zeros((len(initial_belief_state_0), params["n_episodes"]))

        for i, initial_belief in enumerate(initial_belief_state_0):
            params["policy"] = policy
            params["initial_belief_state"] = np.array([initial_belief, 1.0 - initial_belief])

            env = POMDP(params)

            for ep in range(params["n_episodes"]):
                ep_reward = run_episode(env)
                ep_rewards[i, ep] = ep_reward

            # get statistics on the episode rewards
        print("saving")
        np.save(f"output/ep_rewards_{params['policy']}", ep_rewards)


def reward_stats():
    policies = ["greedy_policy", "stochastic_policy", "greedy_value"]
    initial_belief_state_0 = np.arange(0, 1, 0.1)

    for policy in policies:
        Path("output").mkdir(parents=True, exist_ok=True)
        Path("figures").mkdir(parents=True, exist_ok=True)
        ep_rewards = np.load(f"output/ep_rewards_{policy}.npy")
        mean = np.mean(ep_rewards[0])
        std = np.std(ep_rewards[0])
        max_val = np.max(ep_rewards[0])
        min_val = np.min(ep_rewards[0])

        label_str = f"{policy}\n$\mu$ = {round(mean)}, $\sigma$ = {round(std)}\nmax = {round(max_val)}, min = {round(min_val)}"
        sns.distplot(ep_rewards[0], fit=norm, kde=False, label=label_str)
        plt.legend()

    plt.xlabel("Episodic Discounted Reward")
    plt.ylabel("Probability")
    plt.savefig(f"figures/policy_stats.png")


def max_reward():
    rewards = 5 * np.ones(1000)
    gamma = 0.99

    # get discounted gammas for each timestep
    gammas = gamma * np.ones(1000)
    powers = np.arange(0, 1000, step=1)
    discounted_gamma = np.power(gammas, powers)

    # summed discounted rewards
    ep_reward = np.dot(discounted_gamma, rewards)
    print("max reward", ep_reward)


if __name__ == "__main__":
    # main()
    # reward_stats()
    max_reward()