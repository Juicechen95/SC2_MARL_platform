import numpy as np
import tensorflow as tf
from core.Env import make_envs, EnvWrapper
from core.Runner import Runner


class Model:
    def __init__(self, Agt1, Agt2, args):

        num_player = 2
        bot_level = 'easy'
        # bot_level = 'veryeasy', 'easy', 'medium', 'mediumhard', 'hard',
        # 'veryhard', 'cheatvision', 'cheatmoney', 'cheatinsane':

        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.envs = EnvWrapper(make_envs(args, num_player, bot_level))
        # config = self.envs.get_config()
        self.agt1 = Agt1(self.sess)
        self.agt2 = Agt2(self.sess)
        self.sess.run(tf.global_variables_initializer())

        runner = Runner(self.envs, self.agt1, self.agt2, args.steps)
        runner.run()
        
        if args.save_replay:
            self.envs.save_replay('./replays/')
        
        self.envs.close()
        
    