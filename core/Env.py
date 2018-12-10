import os, json
import numpy as np
from s2clientprotocol import ui_pb2 as sc_ui
from s2clientprotocol import spatial_pb2 as sc_spatial
from pysc2.lib import features
from pysc2.lib.actions import FunctionCall, FUNCTIONS, TYPES
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.env.environment import StepType
from pysc2.lib.actions import ABILITY_IDS
from pysc2.lib import actions


class EnvWrapper:
    def __init__(self, envs):
        self.envs = envs

    def step(self, acts):
        act1, act2 = acts
        act1 = self.wrap_actions(act1, self.units1, self.units2)
        act2 = self.wrap_actions(act2, self.units2, self.units1)
        acts = [act1, act2]
        print(acts)
        results = self.envs.step(acts)
        self.update_units_alive(results)
        ts1, ts2 = results
        ob1, rew1, done1 = self.trans_obs(ts1, self.units1, self.units2)
        ob2, rew2, done2 = self.trans_obs(ts2, self.units2, self.units1)
        return [ob1, ob2], [rew1, rew2], [done1, done2]

    def reset(self):
        results = self.envs.reset()
        self.update_unit_list(results)
        self.update_units_alive(results)
        ts1, ts2 = results
        ob1, rew1, done1 = self.trans_obs(ts1, self.units1, self.units2)
        ob2, rew2, done2 = self.trans_obs(ts2, self.units2, self.units1)
        return [ob1, ob2], [rew1, rew2], [done1, done2]

    def wrap_actions(self, actions, alliances, enemies):
        actions = actions[0]
        wrapped_actions = []
        for i in range(len(actions)):
            if alliances[i] not in self.units_alive:
                continue
            wrapped_actions.append(self.mapping(alliances[i], actions[i], enemies))

        return wrapped_actions

    def mapping(self, unit, action, enemies):
        if action == 0:
            return hold(unit)
        elif action <= 4:
            for u in self.feature_units:
                if u.tag == unit:
                    cur_pos = [u.x, u.y]
            unit_dist = [[0, 1], [0, -1], [-1, 0], [1, 0]]
            new_pos = [cur_pos[0] + unit_dist[action - 1][0], cur_pos[1] + unit_dist[action - 1][1]]
            return move(unit, new_pos)
        elif action <= 10:
            target = enemies[action - 5]
            if target in self.units_alive:
                return attack(unit, target)
            else:
                return None

    def update_unit_list(self, timestep):
        ts = timestep[0]
        feature_units = ts.observation.feature_units
        feature_units = sorted(feature_units, key=lambda x: x.tag)
        feature_units = sorted(feature_units, key=lambda x: x.unit_type)
        self.units1 = [u.tag for u in feature_units if u.owner == 1]
        self.units2 = [u.tag for u in feature_units if u.owner == 2]

    def update_units_alive(self, timestep):
        ts = timestep[0]
        self.feature_units = ts.observation.feature_units
        self.units_alive = [u.tag for u in self.feature_units]

    def trans_obs(self, timestep, alliances, enemies):
        ts = timestep
        raw_obs, reward, done = ts.observation, ts.reward, ts.step_type == StepType.LAST
        feature_units = ts.observation.feature_units
        obs_feature = []
        for unit in alliances:
            for u in feature_units:
                if u.tag == unit:
                    obs_feature.append(list(u))  # Define your obs here
                    break
            else:
                obs_feature.append([0]*13)  # details are in pysc2.features.FeatureUnit
        return obs_feature, reward, done

    def save_replay(self, replay_dir='PySC2Replays'):
        self.envs.save_replay(replay_dir)

    def spec(self):
        return self.envs.spec()

    def close(self):
        return self.envs.close()

    @property
    def num_envs(self):
        return self.envs.num_envs


def isalive(unit):
    return unit.health_ratio > 0

args = [
    'screen',
    'minimap',
    'screen2',
    'queued',
    'control_group_act',
    'control_group_id',
    'select_add',
    'select_point_act',
    # 'select_unit_act',
    # 'select_unit_id'
    'select_worker',
    # 'build_queue_id',
    # 'unload_id'
]

defaults = {
            'control_group_act': 0,
            'control_group_id': 0,
            'select_point_act': 0,
            'select_unit_act': 0,
            'select_unit_id': 0,
            'build_queue_id': 0,
            'unload_id': 0,
        }

def hold(u):
    action = sc_pb.Action()
    action.action_raw.unit_command.ability_id = 18
    action.action_raw.unit_command.unit_tags.append(u)
    return action


def move(u, pos):
    action = sc_pb.Action()
    action.action_raw.unit_command.ability_id = 16
    action.action_raw.unit_command.target_world_space_pos.x = pos[0]
    action.action_raw.unit_command.target_world_space_pos.y = pos[1]
    action.action_raw.unit_command.unit_tags.append(u)
    return action


def attack(u, target_unit):
    action = sc_pb.Action()
    action.action_raw.unit_command.ability_id = 23
    action.action_raw.unit_command.target_unit_tag = target_unit
    action.action_raw.unit_command.unit_tags.append(u)
    return action

# def __call__(self, action):
#     defaults = {
#         'control_group_act': 0,
#         'control_group_id': 0,
#         'select_point_act': 0,
#         'select_unit_act': 0,
#         'select_unit_id': 0,
#         'build_queue_id': 0,
#         'unload_id': 0,
#     }
#     fn_id_idx, args = action.pop(0), []
#     fn_id = self.func_ids[fn_id_idx]
#     for arg_type in actions.FUNCTIONS[fn_id].args:
#         arg_name = arg_type.name
#         if arg_name in self.args:
#             arg = action[self.args.index(arg_name)]
#             # pysc2 expects all args in their separate lists
#             if type(arg) not in [list, tuple]:
#                 arg = [arg]
#             # pysc2 expects spatial coords, but we have flattened => attempt to fix
#             if len(arg_type.sizes) > 1 and len(arg) == 1:
#                 arg = [arg[0] % self.spatial_dim, arg[0] // self.spatial_dim]
#             args.append(arg)
#         else:
#             args.append([defaults[arg_name]])
#
#     return [actions.FunctionCall(fn_id, args)]

from pysc2.env import sc2_env
from multiprocessing import Process, Pipe
import collections
from pysc2.env.sc2_env import Agent, Bot


def make_envs(args, num_player, bot_level):
    aif = sc2_env.AgentInterfaceFormat(
                        feature_dimensions=sc2_env.Dimensions(
                        screen=args.sz,
                        minimap=args.sz), use_feature_units=True)
    env_args = dict(map_name=args.map, num_player=num_player, bot_level=bot_level,
                    agent_interface_format=[aif, aif], step_mul=8, game_steps_per_episode=0)
    # return EnvPool([make_env(args.sz, **dict(env_args, visualize=i < args.render)) for i in range(args.envs)])
    return make_env(**dict(env_args, visualize=True))


# based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(num_player, bot_level, **params):

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
        env = sc2_env.SC2Env(players=players, **params)
        return env

    # return _thunk
    return _thunk()


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
