from mytime import now, add_time, runtime
import graphs
import logs
import emails
from environments import Snake, VecSnake, Snake3

from matplotlib import pyplot as plt
import random
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# slow down loading, only when needed:
import tensorflow as tf
from agents import DQNAgent, tensor_reshape, unpack_tensor_reshape, HumanPlayer

# from evaluation import visual_evaluation, score_evaluation
# --------------------------------------------------------------------


# # find true relations between evaluation scores and training scores.
# # used for red rectangles' construction
# score, total_reward, episode_length = logs.read_log('15x15 4 metrics')
# score, total_reward, episode_length = score[1], total_reward[1], episode_length[1]
# iterations = 0
# flag = True
# for game in range(len(episode_length)):
#     iterations += episode_length[game] / 100
#     if flag and iterations > 1000:  # ignore initial memory collection
#         flag = False
#         iterations = 0
#     if iterations > 60000:
#         print('game:', game, '  fits:', iterations)
#         break
