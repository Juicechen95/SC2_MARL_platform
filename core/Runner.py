import time
import numpy as np
from baselines import logger


class Runner:
    def __init__(self, envs, agt1, agt2, n_steps=1000):
        self.state = self.logs = self.ep_rews = None
        self.agt1, self.agt2, self.envs, self.n_steps = agt1, agt2, envs, n_steps

    def run(self, num_updates=100, train=False):
        # based on https://github.com/deepmind/pysc2/blob/master/pysc2/env/run_loop.py
        self.reset()
        try:
            for i in range(num_updates):
                self.logs['updates'] += 1
                rollout = self.collect_rollout()
                if train:
                    self.agt1.train(i, *rollout)
                    self.agt2.train(i, *rollout)
        except KeyboardInterrupt:
            pass
        finally:
            elapsed_time = time.time() - self.logs['start_time']
            frames = 1 * self.n_steps * self.logs['updates']
            print("Took %.3f seconds for %s steps: %.3f fps" % (elapsed_time, frames, frames / elapsed_time))

    def collect_rollout(self):
        states, actions1, actions2 = [[None, None]]*self.n_steps, [[None]*6]*self.n_steps, [[None]*6]*self.n_steps
        rewards1, dones1, values1 = np.zeros((3, self.n_steps, 1))
        rewards2, dones2, values2 = np.zeros((3, self.n_steps, 1))

        for step in range(self.n_steps):
            actions1[step], values1[step] = self.agt1.act(self.state[0])
            actions2[step], values2[step] = self.agt2.act(self.state[1])
            print(actions1[step])
            states[step] = self.state
            actions = [actions1[step], actions2[step]]  # action not taken!
            self.state, rewards, dones = self.envs.step(actions)
            rewards1[step], rewards2[step] = rewards
            dones1[step], dones2[step] = dones

            # we select agt1 as the main player
            self.log(rewards1[step], dones1[step])

        last_value1 = self.agt1.get_value(self.state[0])
        last_value2 = self.agt2.get_value(self.state[1])

        actions = [actions1, actions2]
        rewards = [rewards1, rewards2]
        dones = [dones1, dones2]
        last_value = [last_value1, last_value2]

        return flatten_lists(states[0], num=2), flatten_lists(actions, num=2), rewards, dones, last_value, self.ep_rews

    def reset(self):
        self.state, *_ = self.envs.reset()
        self.logs = {'updates': 0, 'eps': 0, 'rew_best': 0, 'start_time': time.time(),
                     # 'ep_rew': np.zeros(1), 'dones': np.zeros(1)}
                     'ep_rew': np.zeros(1), 'dones': np.zeros(1)}

    def log(self, rewards, dones):
        self.logs['ep_rew'] += rewards
        self.logs['dones'] = np.maximum(self.logs['dones'], dones)
        if sum(self.logs['dones']) < 1:
            return
        self.logs['eps'] += 1
        self.logs['rew_best'] = max(self.logs['rew_best'], np.mean(self.logs['ep_rew']))

        elapsed_time = time.time() - self.logs['start_time']
        frames = 1 * self.n_steps * self.logs['updates']

        self.ep_rews = np.mean(self.logs['ep_rew'])
        logger.logkv('fps', int(frames / elapsed_time))
        logger.logkv('elapsed_time', int(elapsed_time))
        logger.logkv('n_eps', self.logs['eps'])
        logger.logkv('n_samples', frames)
        logger.logkv('n_updates', self.logs['updates'])
        logger.logkv('rew_best_mean', self.logs['rew_best'])
        logger.logkv('rew_max', np.max(self.logs['ep_rew']))
        logger.logkv('rew_mean', np.mean(self.logs['ep_rew']))
        logger.logkv('rew_mestd', np.std(self.logs['ep_rew'])) # weird name to ensure it's above min since logger sorts
        logger.logkv('rew_min', np.min(self.logs['ep_rew']))
        logger.dumpkvs()

        self.logs['dones'] = np.zeros(1)
        self.logs['ep_rew'] = np.zeros(1)


def flatten(x):
    x = np.array(x)  # TODO replace with concat if axis != 0
    return x.reshape(-1, *x.shape[2:])


def flatten_dicts(x):
    return {k: flatten([s[k] for s in x]) for k in x[0].keys()}


#  n-steps x actions x envs -> actions x n-steps*envs
def flatten_lists(input, num=1):
    if num == 1:
        x = input
        output = [flatten([s[a] for s in x]) for a in range(len(x[0]))]
    else:
        output = []
        for i in range(num):
            x = input[i]
            output.append([flatten([s[a] for s in x]) for a in range(len(x[0]))])
    return output
