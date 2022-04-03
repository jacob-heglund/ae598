import numpy as np
import pdb


class Agent:
    def __init__(self, initial_state, initial_belief_state, env):
        self.state = initial_state
        self.belief_state = initial_belief_state
        self.env = env

    def update_state(self, next_state):
        self.state = next_state

    def _belief_state_transition(self, obs):
        # update beliefs of agent based on observations and past beliefs
        #TODO implement this. Shouldn't return anything, but simply updates the values in self.belief_state
        pass

    def act(self, obs):
        # update belief based on observation


        if self.policy == "greedy_policy":
            #TODO this implementation might not be quite right, but the right idea

            ## policy 1: The policy acts according to the most likely belief state, pi(s_b)
            ### where s_b = argmax_{s \ in S} b_t(s)
            most_likely_state = np.argmax(self.belief_state)
            action = env.optimal_policy_full_obs(most_likely_state)

        elif self.policy == "stochastic_policy":
            #TODO this implementation might not be quite right, but the right idea
            ## policy 2: First draw a state as a categorical random variable with probabilities given by the belief state, then act according to the drawn state using the optimal policy, pi(s_b)
            ### where s_b \sim b_t(s) (random draw of state with probabilities given as the belief state probabilities)
            probabilities = self.belief_state
            drawn_state = np.draw_categorical(probabilities)
            action = env.optimal_policy_full_obs(drawn_state)

        elif self.policy == "greedy_value":
            #TODO implement and test
            ## policy 3: need to first compute the optimal Q(s, a) for all state-action pairs assuming fully observable (but still stochastic transitions), then use that as part of the agent's actual policy
            ## we can define the expected action-value as V(a) = \sum_s b_t(s) Q(s, a), which is only a function of action
            ###  we then take the action which maximizes V(a) at each timestep
            pass

        return action


class POMDP:
    def __init__(self, params, gamma=0.99, t_max=1000, seed=0):
        """ 2-state POMDP
        Args:
            agent (Agent): agent for the environment
            t_max (int, optional): max number of episode time steps. Defaults to 1000.
            gamma (float, optional): discount factor for rewards. Defaults to 0.99.
            seed (int, optional): random seed. Defaults to 0.

        """
        self.params = params

        self.agent = Agent(params["initial_state"], params["initial_belief_state"])

        self.state_space = [0, 1]
        self.action_space = [0, 1, 2]
        self.stay_prob = 0.9

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
        self.stay_probability = 0.9
        self.gamma = self.params["gamma"]
        self.rewards = []
        self.t = 0
        self.t_max = self.params["env"]["t_max"]
        self.done = False

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

    def optimal_q_full_obs(self, belief_state, action):
        #TODO compute optimal Q somehow and implement here
        ## I want this to return an action, so just do the averaging V(a) = \sum_s b_t(s) Q(s, a) in here
        ## wait, so what is the state that I choose to use in Q(s, a)? Since I'm summing over all states, I go over all of them.
        pass

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

        return state

    def observation(self, state, action):
        """observation function for POMDP

        Args:
            state (int): s \in {0, 1}, describes current state
            action (int): a \in {0, 1, 2}, describes current action

        Returns:
            int: obs \in {0, 1, -1}, describes observation given previous state and action, -1 is an "error" observation which gives no information about the current state
        """
        if (action == 0) or (action == 1):
            obs = -1
        else:
            obs = state

        return obs

    def reward(self, state, action):
        # agent can't see the immediate reward to make decisions (this is an artificial constraint from Prof. Ornik's definition of the problem), only the discounted episode reward when the episode finishes
        reward = 0
        if state == 0:
            if action == 1:
                reward = 5
            elif action == 2:
                reward = -1
        elif state == 1:
            if action == 1:
                reward = -5
            elif action == 2:
                reward = -1

        return reward

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
        agent.update_state(next_state)
        # get observation at next_state
        obs = self.observation(state, action)
        self.rewards.append(self.reward(state, action))

        self.t += 1

        if self.t >= self.t_max:
            done = True
            ep_reward = self._reward(self.rewards, self.gamma)

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
        initial_state: 0,
        initial_belief_state: np.array([0.5, 0.5]),
        policy: "greedy_policy",
        env: {
            t_max: 1000,
            seed: 0,
            gamma: 0.99
        }
    }
    return params


def run_episode(env):
    # reset to get initial observation

    # while not done, take action, get observation, etc.
    obs = env.reset()

    while not env.done:
        env.agent.act(obs)


def main():
    params = init_params()
    env = POMDP(params)
    for i in range(params["n_episodes"]):
        pdb.set_trace()


if __name__ == "__main__":
    main()