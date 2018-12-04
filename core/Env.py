from pysc2.lib.actions import FunctionCall, FUNCTIONS
from common.config import DEFAULT_ARGS, is_spatial


class EnvWrapper:
    def __init__(self, envs, config):
        self.envs, self.config = envs, config

    def step(self, acts):
        acts = self.wrap_actions(acts)
        results = self.envs.step(acts)
        return self.wrap_results(results)

    def reset(self):
        results = self.envs.reset()
        return self.wrap_results(results)

    def wrap_actions(self, actions):
        acts, args = actions[0], actions[1:]

        wrapped_actions = []
        for i, act in enumerate(acts):
            act_args = []
            for arg_type in FUNCTIONS[act].args:
                act_arg = [DEFAULT_ARGS[arg_type.name]]
                if arg_type.name in self.config.act_args:
                    act_arg = [args[self.config.arg_idx[arg_type.name]][i]]
                if is_spatial(arg_type.name):  # spatial args, convert to coords
                    act_arg = [act_arg[0] % self.config.sz, act_arg[0] // self.config.sz]  # (y,x), fix for PySC2
                act_args.append(act_arg)
            wrapped_actions.append(FunctionCall(act, act_args))

        return wrapped_actions

    def wrap_results(self, results):
        obs = [res.observation for res in results]
        rewards = [res.reward for res in results]
        dones = [res.last() for res in results]

        states = self.config.preprocess(obs)

        return states, rewards, dones

    def save_replay(self, replay_dir='PySC2Replays'):
        self.envs.save_replay(replay_dir)

    def spec(self):
        return self.envs.spec()

    def close(self):
        return self.envs.close()

    @property
    def num_envs(self):
        return self.envs.num_envs


from pysc2.env import sc2_env
from multiprocessing import Process, Pipe


def make_envs(args):
    env_args = dict(map_name=args.map, step_mul=8, game_steps_per_episode=0)
    return EnvPool([make_env(args.sz, **dict(env_args, visualize=i < args.render)) for i in range(args.envs)])


# based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(sz=32, **params):
    def _thunk():
        params['screen_size_px'] = params['minimap_size_px'] = (sz, sz)
        env = sc2_env.SC2Env(**params)
        return env
    return _thunk


# based on https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# SC2Env::step expects actions list and returns obs list so we send [data] and process obs[0]
def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'spec':
            remote.send((env.observation_spec(), env.action_spec()))
        elif cmd == 'step':
            obs = env.step([data])
            remote.send(obs[0])
        elif cmd == 'reset':
            obs = env.reset()
            remote.send(obs[0])
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'save_replay':
            env.save_replay(data)
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class EnvPool(object):
    def __init__(self, env_fns):
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

    def spec(self):
        for remote in self.remotes:
            remote.send(('spec', None))
        results = [remote.recv() for remote in self.remotes]
        # todo maybe support running different envs / specs in the future?
        return results[0]

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        return results

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def save_replay(self, replay_dir='PySC2Replays'):
        self.remotes[0].send(('save_replay', replay_dir))

    @property
    def num_envs(self):
        return len(self.remotes)
