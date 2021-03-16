import numpy as np
import os
from environment import Environment
from driver import Driver
import gym_minigrid
import matplotlib.pyplot as plt

def plotLoss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

def dispatcher(env):

    driver = Driver(env)

    while driver.itr < env.n_train_iters:

        # Train
        if env.train_mode:
            driver.train_step()

        # Test
        if driver.itr % env.test_interval == 0:

            # measure performance
            R = []
            for n in range(env.n_episodes_test):
                if driver.itr == env.n_train_iters-1:
                    R.append(driver.collect_experience(record=True, vis=True, noise_flag=False, n_steps=1000))
                R.append(driver.collect_experience(record=True, vis=env.vis_flag, noise_flag=False, n_steps=1000))

            # update stats
            driver.reward_mean = sum(R) / len(R)
            driver.reward_std = np.std(R)

            # print info line
            driver.print_info_line('full')

            # save snapshot
            if env.train_mode and env.save_models:
                driver.save_model(dir_name=env.config_dir)

        driver.itr += 1

    plotLoss(driver.policy_losses)
    plt.title("Policy Loss")
    plotLoss(driver.disc_losses)
    plt.title("Discriminator Loss")
    plotLoss(driver.forward_losses)
    plt.title("Forward Model Loss")
    plt.show()

if __name__ == '__main__':
    # load environment
    env = Environment(os.path.curdir, 'MiniGrid-FourRooms-v0')

    # start training
    dispatcher(env=env)
