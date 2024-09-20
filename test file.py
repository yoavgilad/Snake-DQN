from mytime import now, wait, add_time, runtime
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


# data30 = logs.read_log('30x30 log')
# data40 = logs.read_log('50x50 log')
# print(data30, '\n', data40)
# data = [['training iterations', 'mean score', '30x30 and 40x40 performance over training period\nenlarged from 22x22'],
#         data30[1], data40[1]]
# fig, ax = plt.subplots()
# labels = data[0]
# ax.set_xlabel(labels[0])
# ax.set_ylabel(labels[1])
# ax.set_title(labels[2])
# xs, ys, label = data[1]
# ax.plot(xs, ys, label=label, color='orange')
# xs, ys, label = data[2]
# ax.plot(xs, ys, label=label, color='red')
# ax.legend()
# plt.savefig('enlargement extent.png')
# plt.show()
# plt.close()
