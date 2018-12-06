import os, json
import numpy as np
from s2clientprotocol import ui_pb2 as sc_ui
from s2clientprotocol import spatial_pb2 as sc_spatial
from pysc2.lib import features
from pysc2.lib.actions import FunctionCall, FUNCTIONS, TYPES
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb


CAT = features.FeatureType.CATEGORICAL

DEFAULT_ARGS = dict(
    screen=0,  # converts to (0,0)
    minimap=0,
    screen2=0,
    queued=False,
    control_group_act=sc_ui.ActionControlGroup.Append,
    control_group_id=1,
    select_point_act=sc_spatial.ActionSpatialUnitSelectionPoint.Select,
    select_add=True,
    select_unit_act=sc_ui.ActionMultiPanel.SelectAllOfType,
    select_unit_id=0,
    select_worker=sc_ui.ActionSelectIdleWorker.AddAll,
    build_queue_id=0,
    unload_id=0
)

# name => dims
# single player version:
# NON_SPATIAL_FEATURES = dict(
#     player=(11,),
#     game_loop=(1,),
#     score_cumulative=(13,),
#     available_actions=(len(FUNCTIONS),),
#     single_select=(1, 7),
#     # multi_select=(0, 7), # TODO
#     # cargo=(0, 7), # TODO
#     cargo_slots_available=(1,),
#     # build_queue=(0, 7), # TODO
#     control_groups=(10, 2),
# )

NON_SPATIAL_FEATURES = dict(
    player=(11,),
    game_loop=(1,),
    score_cumulative=(13,),
    available_actions=(len(FUNCTIONS),),
    single_select=(1, 7),
    # multi_select=(0, 7), # TODO
    # cargo=(0, 7), # TODO
    cargo_slots_available=(1,),
    # build_queue=(0, 7), # TODO
    control_groups=(10, 2),
)


class EnvWrapper:
    def __init__(self, envs, args):
        self.envs = envs
        self.config = Config(args.sz, args.map, args.run_id)

    def get_config(self):
        return self.config

    def step(self, acts):
        act1, act2 = acts
        act1 = self.wrap_actions(act1)
        act2 = self.wrap_actions(act2)
        acts = [act1, act2]
        results = self.envs.step(acts)
        ob1, ob2 = results
        ob1 = self.wrap_results(ob1)
        ob2 = self.wrap_results(ob2)
        return [ob1, ob2]

    def reset(self):
        results = self.envs.reset()
        ob1, ob2 = results
        ob1 = self.wrap_results(ob1)
        ob2 = self.wrap_results(ob2)
        return [ob1, ob2]

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
import collections
Agent = collections.namedtuple("Agent", ["race"])
Bot = collections.namedtuple("Bot", ["race", "difficulty"])


def make_envs(args, num_player, bot_level):
    env_args = dict(map_name=args.map, num_player=num_player, bot_level=bot_level, step_mul=8, game_steps_per_episode=0)
    return EnvPool([make_env(args.sz, **dict(env_args, visualize=i < args.render)) for i in range(args.envs)])


# based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(num_player, bot_level, sz=32, **params):

    if num_player == 2:
        players = [Agent(sc_common.Random), Agent(sc_common.Random)]  # race here only affects the type of UI
    elif num_player == 1:
        if bot_level == 'veryeasy':
            bot = Bot(sc_common.Random, sc_pb.VeryEasy)
        elif bot_level == 'easy':
            bot = Bot(sc_common.Random, sc_pb.Easy)
        elif bot_level == 'medium':
            bot = Bot(sc_common.Random, sc_pb.Medium)
        elif bot_level == 'mediumhard':
            bot = Bot(sc_common.Random, sc_pb.MediumHard)
        elif bot_level == 'hard':
            bot = Bot(sc_common.Random, sc_pb.Hard)
        elif bot_level == 'veryhard':
            bot = Bot(sc_common.Random, sc_pb.VeryHard)
        elif bot_level == 'cheatvision':
            bot = Bot(sc_common.Random, sc_pb.CheatVision)
        elif bot_level == 'cheatmoney':
            bot = Bot(sc_common.Random, sc_pb.CheatMoney)
        elif bot_level == 'cheatinsane':
            bot = Bot(sc_common.Random, sc_pb.CheatInsane)
        else:
            raise ValueError
        players = [Agent(sc_common.Random), bot]
    else:
        raise ValueError

    def _thunk():
        params['screen_size_px'] = params['minimap_size_px'] = (sz, sz)
        env = sc2_env.SC2Env(players=players, **params)
        return env

    return _thunk


# class Difficulty(enum.IntEnum):
#   """Bot difficulties."""
#   very_easy = sc_pb.VeryEasy
#   easy = sc_pb.Easy
#   medium = sc_pb.Medium
#   medium_hard = sc_pb.MediumHard
#   hard = sc_pb.Hard
#   harder = sc_pb.Harder
#   very_hard = sc_pb.VeryHard
#   cheat_vision = sc_pb.CheatVision
#   cheat_money = sc_pb.CheatMoney
#   cheat_insane = sc_pb.CheatInsane


# based on https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# SC2Env::step expects actions list and returns obs list so we send [data] and process obs[0]
def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'spec':
            remote.send((env.observation_spec(), env.action_spec()))
        elif cmd == 'step':
            obs = env.step(data)
            remote.send(obs)
        elif cmd == 'reset':
            obs = env.reset()
            remote.send(obs)
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


class Config:
    # TODO extract embed_dim_fn to config
    def __init__(self, sz, map, run_id, embed_dim_fn=lambda x: max(1, round(np.log2(x)))):
        self.run_id = run_id
        self.sz, self.map = sz, map
        self.embed_dim_fn = embed_dim_fn
        self.feats = self.acts = self.act_args = self.arg_idx = self.ns_idx = None
        cfg_path = 'res/config.json.dist'
        self.build(cfg_path)

    def build(self, cfg_path):
        feats, acts, act_args = self._load(cfg_path)

        if 'screen' not in feats:
            feats['screen'] = features.SCREEN_FEATURES._fields
        if 'minimap' not in feats:
            feats['minimap'] = features.MINIMAP_FEATURES._fields
        if 'non_spatial' not in feats:
            feats['non_spatial'] = NON_SPATIAL_FEATURES.keys()
        self.feats = feats

        # TODO not connected to anything atm
        if acts is None:
            acts = FUNCTIONS
        self.acts = acts

        if act_args is None:
            act_args = TYPES._fields
        self.act_args = act_args

        self.arg_idx = {arg: i for i, arg in enumerate(self.act_args)}
        self.ns_idx = {f: i for i, f in enumerate(self.feats['non_spatial'])}

        print('feats')
        print(self.feats)
        print('arg_idx')
        print(self.arg_idx)
        print('ns_idx')
        print(self.ns_idx)
        print('NON_SPATIAL_FEATURES')
        print(NON_SPATIAL_FEATURES)

    def map_id(self):
        return self.map + str(self.sz)

    def full_id(self):
        if self.run_id == -1:
            return self.map_id()
        return self.map_id() + "/" + str(self.run_id)

    def policy_dims(self):
        return [(len(self.acts), 0)] + [(getattr(TYPES, arg).sizes[0], is_spatial(arg)) for arg in self.act_args]

    def screen_dims(self):
        return self._dims('screen')

    def minimap_dims(self):
        return self._dims('minimap')

    def non_spatial_dims(self):
        return [NON_SPATIAL_FEATURES[f] for f in self.feats['non_spatial']]

    # TODO maybe move preprocessing code into separate class?
    def preprocess(self, obs):
        return [self._preprocess(obs, _type) for _type in ['screen', 'minimap'] + self.feats['non_spatial']]

    def _dims(self, _type):
        return [f.scale ** (f.type == CAT) for f in self._feats(_type)]

    def _feats(self, _type):
        feats = getattr(features, _type.upper() + '_FEATURES')
        return [getattr(feats, f_name) for f_name in self.feats[_type]]

    def _preprocess(self, obs, _type):
        if _type in self.feats['non_spatial']:
            return np.array([self._preprocess_non_spatial(ob, _type) for ob in obs])
        spatial = [[ob[_type][f.index] for f in self._feats(_type)] for ob in obs]
        return np.array(spatial).transpose((0, 2, 3, 1))

    def _preprocess_non_spatial(self, ob, _type):
        if _type == 'available_actions':
            acts = np.zeros(len(self.acts))
            acts[ob['available_actions']] = 1
            return acts
        return ob[_type]

    def save(self, cfg_path):
        with open(cfg_path, 'w') as fl:
            json.dump({'feats': self.feats, 'act_args': self.act_args}, fl)

    def _load(self, cfg_path):
        with open(cfg_path, 'r') as fl:
            data = json.load(fl)
        return data.get('feats'), data.get('acts'), data.get('act_args')


def is_spatial(arg):
    return arg in ['screen', 'screen2', 'minimap']
