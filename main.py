import numpy as np
import os
from environment import Environment
from driver import Driver
import gym_minigrid
import pybulletgym
import matplotlib.pyplot as plt

def plotLoss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

def plotReward(rewards, stds=None):
    plt.figure()
    plt.plot(rewards)
    # if stds != None and len(stds) > 0:
    #     plt.errorbar(list(range(len(rewards))), rewards, yerr=stds, ecolor='r')
    plt.xlabel('Epochs')
    plt.ylabel('Avg Reward')

def dispatcher(env, use_irl, env_name='Hopper'):

    driver = Driver(env, use_irl)
    avg_rewards = []
    reward_stds = []
    if env.vis_flag:
        env.render()

    while driver.itr < env.n_train_iters:

        # Train
        if env.train_mode:
            driver.train_step()

        # Test
        if driver.itr % env.test_interval == 0:

            # measure performance
            R = []
            for n in range(env.n_episodes_test):
                R.append(driver.collect_experience(record=True, vis=env.vis_flag, noise_flag=False, n_steps=1000))

            # update stats
            driver.reward_mean = sum(R) / len(R)
            driver.reward_std = np.std(R)
            avg_rewards.append(driver.reward_mean)
            reward_stds.append(driver.reward_std)

            # print info line
            driver.print_info_line('full')

            # save snapshot
            if env.train_mode and env.save_models:
                driver.save_model(dir_name=env.config_dir)

        driver.itr += 1

    print('Top 3 Max Avg Rewards: ', np.sort(avg_rewards)[-3:])
    plotLoss(driver.policy_losses)
    plt.title("Policy Loss")
    plotLoss(driver.disc_losses)
    plt.title("Discriminator Loss")
    plotLoss(driver.forward_losses)
    plt.title("Forward Model Loss")
    plotReward(avg_rewards, reward_stds)
    plt.title(env_name + " Average Rewards")
    plt.show()

if __name__ == '__main__':
    # load environment
    env_name = 'Hopper' # options: Hopper, Ant, ...
    env = Environment(os.path.curdir, env_name + 'PyBulletEnv-v0')
    use_irl = True
    # start training
    dispatcher(env=env, use_irl=use_irl, env_name=env_name)
