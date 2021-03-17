from garage.experiment import Snapshotter
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session

import numpy as np
import h5py
import argparse



MODEL_PATH = "data/local/experiment/trpo_minigrid_115/"

# MAIN IDEA: https://github.com/rlworkgroup/garage/blob/c43eaf7647f7feb467847cb8bc107301a7c31938/docs/user/reuse_garage_policy.md

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/pos': [],
            'infos/orientation': [],
            }

def append_data(data, s, a, tgt, done, pos, ori, rew):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(rew)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/pos'].append(pos)
    data['infos/orientation'].append(ori)

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32
        data[k] = np.array(data[k], dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--num_episodes', type=int, default=int(1e5), help='Num episodes to collect')
    args = parser.parse_args()


    buffer_data = reset_data()
    snapshotter = Snapshotter()

    with tf.compat.v1.Session(): # optional, only for TensorFlow
        data = snapshotter.load(MODEL_PATH)
        policy = data['algo'].policy
        env = data['env']

        for n_s in range(args.num_episodes):
            obs = env.reset()  # The initial observation
            policy.reset()
            done = False
            rew = 0.0
            if n_s % 1000 == 0:
                print('episode: ', n_s)
            
            # max_step is preset inside the env, so done will be True to break the loop when
            # either max_step is reached or goal achieved
            while True:
                if args.render:
                    env.render()  # Render the environment to see what's going on (optional)

                act = policy.get_action(obs)
                
                # act[0] is the actual action, while the second tuple is the done variable. Inspiration: 
                # https://github.com/lcswillems/rl-starter-files/blob/3c7289765883ca681e586b51acf99df1351f8ead/utils/agent.py#L47
                append_data(buffer_data, obs, act[0], None, done, env.agent_pos, env.agent_dir, rew) # obs is flattened from 7*7*2
                new_obs, rew, done, _ = env.step(act[0]) # why [0] ?

                if done: 
                    # reset target here!
                    # can randomly sample an action because the action at terminal state will be discarded anyway.
                    append_data(buffer_data, new_obs, env.action_space.sample(), None, done, env.agent_pos, env.agent_dir, rew)
                    break

                else:
                    # continue by setting current obs
                    obs = new_obs

        fname = 'minigrid4rooms_generated_115.hdf5'
        dataset = h5py.File(fname, 'w')
        npify(buffer_data)
        for key in buffer_data:
            dataset.create_dataset(key, data=buffer_data[key], compression='gzip')

        env.close()

if __name__ == "__main__":
    main()




