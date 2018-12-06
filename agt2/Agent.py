import tensorflow as tf
from tensorflow.contrib import layers
from agt1.Config import Config


# The following codes are for agent vs agent scenario where two different agents are needed
# If you only need to use one agent (in agt1) and hope to set agt2 to built-in AI,
# please follow these steps:
# 1. Set num_player = 1 and choose a value for bot_level in Model.py
# 2. Remove all vars ended with 2 (indicating player2)


Config = Config()

class Agent:
    def __init__(self, sess, default_config):
        """
        :param Config:
            Config.restore: whether to restore a saved policy model
            Config.restore_path: path to the saved policy model of agent, ./model/ by default.
        """
        self.sess = sess
        self.config = default_config

        # set up policy network
        # under specific scope (recommend 'pol2')
        with tf.variable_scope('pol2'):
            self.policy = self.model_fn(Config)  # example policy

            # restore model if provided
            # model stored in ./model/model.ckpt.meta
            # restore from scope('pol2') to scope('pol2')
            if Config.restore:
                self.pol_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pol2')
                self.pol_saver = tf.train.Saver(self.pol_vars)
                self.pol_saver.restore(self.sess, Config.restore_path)
            
        def sample(probs):
            u = tf.random_uniform(tf.shape(probs))
            return tf.argmax(tf.log(u) / probs, axis=1)

        self.action = [sample(p) for p in self.policy]

    def model_fn(self, config):
        from pysc2.lib.actions import FUNCTIONS, TYPES  # actions
        acts = FUNCTIONS
        act_args = TYPES._fields
        """
        ('screen', 'minimap', 'screen2', 'queued', 'control_group_act', 'control_group_id', 
        'select_point_act', 'select_add', 'select_unit_act', 'select_unit_id', 'select_worker',
        'build_queue_id', 'unload_id')
        """
        def is_spatial(arg):
            return arg in ['screen', 'screen2', 'minimap']  # all spatial args

        def cnn_block(sz, dims, embed_dim_fn):
            block_input = tf.placeholder(tf.float32, [None, sz, sz, len(dims)])
            block = tf.transpose(block_input, [0, 3, 1, 2])  # NHWC -> NCHW
            block = tf.split(block, len(dims), axis=1)
            for i, d in enumerate(dims):
                if d > 1:
                    block[i] = tf.one_hot(tf.to_int32(tf.squeeze(block[i], axis=1)), d, axis=1)
                    block[i] = layers.conv2d(block[i], num_outputs=embed_dim_fn(d), kernel_size=1, data_format="NCHW")
                else:
                    block[i] = tf.log(block[i] + 1.0)
            block = tf.concat(block, axis=1)

            conv1 = layers.conv2d(block, num_outputs=16, kernel_size=5, data_format="NCHW")
            conv2 = layers.conv2d(conv1, num_outputs=32, kernel_size=3, data_format="NCHW")
            return conv2

        def non_spatial_block(sz, dims, idx):
            block_inputs = [tf.placeholder(tf.float32, [None, *dim]) for dim in dims]
            block = broadcast(tf.log(block_inputs[idx['player']] + 1.0), sz)
            return block

        def broadcast(tensor, sz):
            return tf.tile(tf.expand_dims(tf.expand_dims(tensor, 2), 3), [1, 1, sz, sz])

        screen = cnn_block(self.config.sz, self.config.screen_dims(), self.config.embed_dim_fn)
        minimap = cnn_block(self.config.sz, self.config.minimap_dims(), self.config.embed_dim_fn)
        non_spatial = non_spatial_block(self.config.sz, self.config.non_spatial_dims(), self.config.ns_idx)

        state = tf.concat([screen, minimap, non_spatial], axis=1)
        fc1 = layers.fully_connected(layers.flatten(state), num_outputs=256)

        policy = []
        for arg in act_args:
            if is_spatial(arg):
                logits = layers.conv2d(state, num_outputs=1, kernel_size=1, activation_fn=None, data_format="NCHW")
                policy.append(tf.nn.softmax(layers.flatten(logits)))
            else:
                policy.append(layers.fully_connected(fc1, num_outputs=getattr(TYPES, arg).sizes[0], activation_fn=tf.nn.softmax))

        return policy  # and something else like value if you choose A2C

    def act(self, state):
        return self.sess.run(self.action, feed_dict=dict(state))

    def train(self, n_update, *rollout):
        # train agt here (example in https://github.com/inoryy/reaver-pysc2/blob/v1.0/rl/agent.py),
        # or in Model.py (example in https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py)
        pass

