from mytime import now, runtime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import models
from evaluation import score_evaluation, visual_evaluation
from agents import HumanPlayer, RandomPlayer

# only modify this segment
# ------------------------------------------------------------------------
model_name = '30x30 4'
render = True  # vec_size = episodes if render = False
fps = 0
episodes = 1000  # -1 for endless evaluation. only works if render = True
loop_threshold = 200
foods = (1, 1)
start_lengths = (4, 4)
mask = False
old_actions = False
# ------------------------------------------------------------------------

match model_name:
    case 'random':
        model = RandomPlayer((1, 22, 22, 1))
    case 'human':
        model = HumanPlayer((1, 30, 30, 1))
    case _:
        model = models.load_model(model_name + '.keras')

start_time = now()
if render:
    print('mean score: ',
          visual_evaluation(model, episodes, loop_threshold, foods, start_lengths, fps, old_actions, mask))
else:
    print('mean score: ',
          score_evaluation(model, episodes, loop_threshold, foods, start_lengths, True, model_name + ' distribution',
                           old_actions=old_actions))

print(f'runtime is {runtime(start_time)}')
