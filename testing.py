from mytime import now, runtime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import models
from evaluation import score_evaluation, visual_evaluation
from agents import HumanPlayer, RandomPlayer

# Deployment settings - only modify this segment
# ------------------------------------------------------------------------------------
model_name = '15x15 4'  # 'random' for random player, 'human' for human player
render = True  # If render = False then all episodes will be played in parallel
fps = 30  # FPS limit; may run slower for extremely large models
episodes = 1000  # Amount of games. -1 for endless evaluation, requires render=True
loop_threshold = -1  # Negative value for no loop detection
foods = (1, 1)  # A random int in this range is chosen each episode
start_lengths = (4, 4)  # A random int in this range is chosen each episode
save_best = True  # Whether to save the best game
old_actions = False  # True = relative movement, False = absolute movement
mask = True  # Whether to mask instantly-losing actions
height, width = 16, 16  # Only affects human & random player
# ------------------------------------------------------------------------------------

# Load model
match model_name.lower():
    case 'random':
        model = RandomPlayer((1, height, width, 1))
    case 'human':
        model = HumanPlayer((1, height, width, 1))
    case _:
        model = models.load_model(model_name + '.keras')

# Save timestamp for runtime calculation
start_time = now()
# Evaluate
if render:
    mean_score = visual_evaluation(model, episodes, loop_threshold, foods, start_lengths,
                                   fps, save_best, old_actions, mask)
else:
    mean_score = score_evaluation(model, episodes, loop_threshold, foods, start_lengths,
                                  save_best, model_name + ' distribution', old_actions)
# Summarize
print('mean score: ', mean_score)
print(f'runtime was {runtime(start_time)}')
