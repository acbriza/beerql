
import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf

from lib import plotting from lib
from collections import deque, namedtuple
from dqn import *

""" Code lifted and adapted from https://github.com/dennybritz/reinforcement-learning
"""


def testing1():
    tf.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)

    e = Estimator(scope="test")
    sp = StateProcessor()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Example observation batch
        observation = env.reset()
        
        observation_p = sp.process(sess, observation)
        observation = np.stack([observation_p] * 4, axis=2)
        observations = np.array([observation] * 2)
        
        # Test Prediction
        print(e.predict(sess, observations))

        # Test training step
        y = np.array([10.0, 10.0])
        a = np.array([1, 3])
        print(e.update(sess, observations, a, y))



def testing2():
    tf.reset_default_graph()

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)
        
    # Create estimators
    q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
    target_estimator = Estimator(scope="target_q")

    # State processor
    state_processor = StateProcessor()

    # Run it!
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t, stats in deep_q_learning(sess,
                                        env,
                                        q_estimator=q_estimator,
                                        target_estimator=target_estimator,
                                        state_processor=state_processor,
                                        experiment_dir=experiment_dir,
                                        num_episodes=10000,
                                        replay_memory_size=500000,
                                        replay_memory_init_size=50000,
                                        update_target_estimator_every=10000,
                                        epsilon_start=1.0,
                                        epsilon_end=0.1,
                                        epsilon_decay_steps=500000,
                                        discount_factor=0.99,
                                        batch_size=32):

            print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))