import numpy as np
import tensorflow as tf
from core.Env import make_envs, EnvWrapper
from core.Runner import Runner


class Model:
    def __init__(self, Agt1, Config1, Agt2, Config2, args):
        tf.reset_default_graph()
        self.sess = tf.Session()

        self.envs = EnvWrapper(make_envs(args))
        agt1 = Agt1(Config1)
        agt2 = Agt2(Config2)

        # restore model

        runner = Runner(self.envs, agt1, agt2)
        runner.run()
        
        if args.save_replay:
            self.envs.save_replay('./replays/')
        
        self.envs.close()
        
    