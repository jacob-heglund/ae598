import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb


def plotting():
    os.makedirs("figures", exist_ok=True)

    policies=[1, 2, 3, 4, 5, 6, 9]
    fig, ax = plt.subplots(ncols=2)

    for policy in policies:
        print(policy)
        cumulative_episode_rewards = np.load(f"output/episode_reward_policy_{policy}.npy")
        times = np.arange(1, cumulative_episode_rewards.shape[1]+1)

        # sum of rewards up to time it was recorded
        # you don't need to do this sum here b/c you already sum in the episode!
        ## don't do that in the future! You want as close to raw numbers as possible
        # cumulative_reward = np.cumsum(episode_rewards, axis=1)

        # average each episode over the amount of time that passed up to the point the reward was recorded
        time_average_cumulative_reward = cumulative_episode_rewards / times

        # then average over all episodes to get time-averaged, average cumulative sum at each time step across episodes
        episode_average_cumulative_reward = np.mean(time_average_cumulative_reward, axis=0)

        # compute epirical variance across episodes
        mean = np.expand_dims(episode_average_cumulative_reward, 1)
        mean = np.tile(mean, 10000)
        mean = mean.transpose()
        empirical_variance =  np.sum((time_average_cumulative_reward - mean) ** 2, axis=0) / (mean.shape[0])

        if policy == 7 or policy == 8 or policy == 9:
            if policy == 7:
                action = 0
            elif policy == 8:
                action = 1
            elif policy == 9:
                action = 2
            label_str = f"Policy 7 - Action {action}"
        else:
            label_str = f"Policy {policy}"

        # plot 1 - plot empirical mean o summed return up to current time step of simulation
        ax_curr = ax[0]
        ax_curr.plot(times, episode_average_cumulative_reward, label=label_str, alpha=0.5)
        ax_curr.set_xlabel("Episode Run Time")
        ax_curr.set_ylabel("Empirical Mean of Rewards Averaged over Episodes")
        ax_curr.legend()

        # plot 2 - plot empirical variance of summed return up to current time step of simulation
        ax_curr = ax[1]
        ax_curr.plot(times, empirical_variance, label=label_str, alpha=0.5)
        ax_curr.set_xlabel("Episode Run Time")
        ax_curr.set_ylabel("Empirical Variance of Rewards Averaged over Episodes")
        ax_curr.legend()

    plt.tight_layout()
    plt.savefig(f"figures/reward_statistics.png")


    """
    takeaways

    Policies 1 and 2 are basically the same in terms of mean performance across episodes. This makes sense, as they are basically the same policy, one just explores a little more before commiting. During the exploration, it might get worse rewards (it is only a 1000 length episode), but seems to make that up almost exactly for the rest of the episode.

        For some reason, policy 4 is better than policy 3.

        For some reason, policy 5 does worse than policy 4. This shouldn't be the case and I think something is wrong
    """


if __name__ == "__main__":
    plotting()