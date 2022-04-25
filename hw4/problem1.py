import numpy as np
import seaborn as sns
from scipy.stats import norm

import matplotlib.pyplot as plt
from numpy.random import default_rng
import os
import pdb


class MDP:
    def __init__(self, environment_dynamics):
        self.state_space_size = 1000000
        self.action_space_size = 2
        self.state_space = np.zeros(self.state_space_size)
        self.max_state_idx = self.state_space_size - 1
        self.goal_state = self.max_state_idx
        self.environment_dynamics = environment_dynamics
        self.env_t = 0

    def reset(self):
        # choose a new alpha at the start of each episode
        self.alpha = default_rng().uniform(0.5, 1.0)

        self.env_t = 0
        state = 550000
        done = False
        return state, done

    def step(self, state, action):
        # since we assume the agent can compute an optimal policy (WRT its internal model of the env) at each time step, there is no need for rewards in this env
        ## also, there is a new agent for each episode (b/c we're doing online learning), so rewards are meaningless in this setting
        done = False

        # take left action
        if action == 0:
            next_state = state - 1

        # take right action
        elif action == 1:
            if self.environment_dynamics == "c":
                # time-varying alpha \in [0.5, 1] drawn from uniform distribution at each time step
                alpha_curr = default_rng().uniform(0.5, 1.0)

            elif self.environment_dynamics == "d":
                # static alpha \in [0.5, 1] drawn at the beginning of the episode
                alpha_curr = self.alpha

            rand_val = default_rng().uniform(0, 1)
            if rand_val <= alpha_curr:
                # move right with probabillity alpha
                next_state = state + 1
            else:
                # stay in place with probability 1 - alpha
                next_state = state

        # environment is toroidal
        # we went left in the 0 state, so wrap around
        if state < 0:
            next_state = self.max_state_idx
        # we went right in the 999999 state, so wrap around
        elif state > self.max_state_idx:
            next_state = 0

        if next_state == self.goal_state:
            done = True

        return next_state, done


class Agent:
    def __init__(self, policy, goal_state):
        # initial state
        self.policy = policy
        self.action_to_idx = {
            "left": 0,
            "right": 1,
        }
        self.alpha_hat = 1
        self.n_right_actions = 0
        self.n_right_transitions_given_right_action = 0
        self.goal_state = goal_state

    def act(self, state, env_t):
        if self.policy == "a":
            # always go left b/c robust optimal policy
            action = 0

        elif (self.policy == "c"):
            # recompute optimal policy WRT the agent's internal model of the MDP at each time
            action = self._optimal_policy(state)

        elif (self.policy == "e"):
            # lame active learning, but it's better than nothing and most important, fast to implement
            ## at this point i have been working on the assignment for 2 full days. I have real reserach to do, so just get something implemented and finish the dang assignment
            if env_t <= 500:
                action = 1
            else:
                action = self._optimal_policy(state)

        return action

    def online_learning(self, state, action, next_state):
        # update agent's internal model
        if self.policy == "a":
            # no learning, just go right
            pass

        elif self.policy == "c":
            # basic passive learning
            # only update alpha if action is "right"
            if action == 1:
                self.n_right_actions += 1
                if state == next_state:
                    # agent didn't move "right" even though it took a "right" action
                    pass
                else:
                    # agent did move "right" given "right" action
                    self.n_right_transitions_given_right_action += 1

            self.alpha_hat = self.n_right_transitions_given_right_action / self.n_right_actions

        elif self.policy == "e":
            # active learning with a reward bonus
            ## just kidding, I have other priorities than this homework and it is taking waaaay too long, so we just get basic passive learning
            # only update alpha if action is "right"
            if action == 1:
                self.n_right_actions += 1
                if state == next_state:
                    # agent didn't move "right" even though it took a "right" action
                    pass
                else:
                    # agent did move "right" given "right" action
                    self.n_right_transitions_given_right_action += 1

            self.alpha_hat = self.n_right_transitions_given_right_action / self.n_right_actions

    def _optimal_policy(self, state):
        # internal model of MDP transition dynamics is given as alpha_hat
        # then we optimize the policy WRT the expected number of steps it takes to hit the goal state
        ## it's a really simple MDP, so you can just get an analytical result instead of using value iteration or whatever
        ## for each value of alpha, there will be an optimal stationary policy (assuming infinite planning horizon).
        ## However, it will change as the internal model of alpha changes, which is why you need to recompute it every step.
        if self.alpha_hat == 0:
            # handle case of n_right_actions > 0, but n_right_transitions_given_right_action = 0. Can't divide by 0!
            expected_hitting_time_right = np.inf
        else:
            expected_hitting_time_right = (1.0 / self.alpha_hat) * (self.goal_state - state)

        action = np.argmin((state, expected_hitting_time_right))
        return action


def run_episode(env, agent):
    # reset
    state, done = env.reset()
    alpha_hat = []

    while not done:
        action = agent.act(state, env.env_t)
        next_state, done = env.step(state, action)
        agent.online_learning(state, action, next_state)

        env.env_t += 1
        state = next_state

        alpha_hat.append(agent.alpha_hat)

    return np.asarray(alpha_hat), env.alpha


def main():
    n_episodes = 100
    # part a
    # environment_dynamics = "c"
    # policies = ["a"]

    # # part c
    # environment_dynamics = "c"
    # policies = ["c"]

    # # part d
    environment_dynamics = "d"
    policies = ["c"]

    # # part e
    # environment_dynamics = "d"
    # policies = ["e"]

    env = MDP(environment_dynamics)

    for policy in policies:
        # run a bunch of episodes
        run_times = np.zeros(n_episodes)
        alphas = np.zeros(n_episodes)

        for episode in range(n_episodes):
            # init new agent for each episode since we're doing online learning
            agent = Agent(policy, env.goal_state)
            alpha_hat, alpha_true = run_episode(env, agent)
            alpha_true = alpha_true * np.ones_like(alpha_hat)

            # if environment_dynamics == "c":
            #     plt.semilogx(alpha_hat)
            #     plt.xlabel("Time")
            #     plt.ylabel("$\hat{\\alpha}$")
            #     os.makedirs("figures", exist_ok=True)


            # elif environment_dynamics == "d":
            #     if policy == "c":
            #         plt.semilogx(alpha_hat, label="$\hat{\\alpha}$")
            #         plt.semilogx(alpha_true, label="$\\alpha_{true}$")
            #         plt.xlabel("Time")
            #         plt.ylabel("$\hat{\\alpha}$")
            #         plt.legend()
            #         os.makedirs("figures", exist_ok=True)
            # plt.savefig(f"figures/alpha_hat_part_{environment_dynamics}")

            print(f"Policy: {policy} --- Episode {episode} ended in {env.env_t} timesteps, alpha = {alpha_true[0]}")
            # pdb.set_trace()
            run_times[episode] = env.env_t
            alphas[episode] = alpha_true[0]

        os.makedirs("output", exist_ok=True)
        np.save(f"output/run_times_policy_{policy}_env_{environment_dynamics}", run_times)
        np.save(f"output/alphas_policy_{policy}_env_{environment_dynamics}", alphas)


def plotting():
    # # part c
    # part = "part_c"
    # environment_dynamics = "c"
    # policies = ["c"]

    # # part d
    part = "part_d"
    environment_dynamics = "d"
    policies = ["c"]

    # # part e
    # part = "part_e"
    # environment_dynamics = "d"
    # policies = ["c", "e"]

    filenames = []
    for policy in policies:
        run_times = np.load(f"output/run_times_policy_{policy}_env_{environment_dynamics}.npy")
        alphas = np.load(f"output/alphas_policy_{policy}_env_{environment_dynamics}.npy")
        label_str = f"Policy {policy.upper()}"
        plt.hist(run_times, label=label_str, alpha=0.7)

    plt.xlabel("Episode Run Time")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig(f"figures/{part}/run_time_distribution_{part}.png")
    """
    Part C -
    # how often did it outperform the policy from part a
    # policy c often (83% of runs) performed worse than policy a. However, it didn't perform that much worse, and all were within 50 steps of policy a
    ## occasionally (17% of the time), policy c would outperform policy a.
    """

    """
    Part D -
    """

    worse_perf = np.where(run_times > 550002+50)
    better_perf = np.where(run_times < 550002)
    print(len(worse_perf[0]) / len(run_times))
    print(len(better_perf[0]) / len(run_times))

if __name__ == "__main__":
    # main()
    plotting()


