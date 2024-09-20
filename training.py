import logs
import emails
import graphs
from environments import VecSnake, VecSnake3
from agents import DQNAgent, tensor_reshape
from evaluation import score_evaluation
from mytime import now, add_time, runtime

import random
import numpy as np
from numpy.typing import NDArray
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import multiprocessing as mp

import cProfile
import pstats
import io

profile: bool = False
multi_processing: bool = False  # causes retracing when pygame is present
save_log: bool = True
save_metrics: bool = True
save_graph: bool = True
verbose: bool = True
summary_email: bool = True
live_emails: bool = False

MODEL_NAME = '15x15 4'
OLD_MODEL_NAME = ''
parameters = \
    {
        '12x12': {'OLD_ACTIONS': True,
                  'HEIGHT': 12,
                  'WIDTH': 12,
                  'MIN_FOODS': 1,
                  'MAX_FOODS': 1,
                  'MIN_START_LENGTH': 4,
                  'MAX_START_LENGTH': 4,

                  'FITS': 10000,
                  'ENVS': 5,
                  'FIT_FREQ': 20,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 100_000,
                  'TARGET_SYNC_FREQ': 100,
                  'EPOCHS_PER_FIT': 1,

                  'GAMMA': 0.95,
                  'TAU': 1,
                  'EPSILON': 0.5,
                  'EPSILON_DECAY': 0.999,
                  'MIN_EPSILON': 0.05,
                  'DECAY_BEFORE_TRAINING': False,

                  'PATIENCE': 200,
                  'PATIENCE_MEMORY_SIZE': 100,
                  'STOP_EARLY': False, },
        '13x13': {'OLD_ACTIONS': True,
                  'HEIGHT': 13,
                  'WIDTH': 13,
                  'MIN_FOODS': 1,
                  'MAX_FOODS': 1,
                  'MIN_START_LENGTH': 4,
                  'MAX_START_LENGTH': 4,

                  'FITS': 1000,
                  'ENVS': 5,
                  'FIT_FREQ': 30,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 100_000,
                  'TARGET_SYNC_FREQ': 100,
                  'EPOCHS_PER_FIT': 1,

                  'GAMMA': 0.95,
                  'TAU': 1,
                  'EPSILON': 0.01,
                  'EPSILON_DECAY': 0.9975,
                  'MIN_EPSILON': 0.01,
                  'DECAY_BEFORE_TRAINING': False,

                  'PATIENCE': 200,
                  'PATIENCE_MEMORY_SIZE': 100,
                  'STOP_EARLY': False, },
        '13x13 duel': {'OLD_ACTIONS': True,
                       'HEIGHT': 13,
                       'WIDTH': 13,
                       'MIN_FOODS': 1,
                       'MAX_FOODS': 1,
                       'MIN_START_LENGTH': 4,
                       'MAX_START_LENGTH': 4,

                       'FITS': 1000,
                       'ENVS': 5,
                       'FIT_FREQ': 30,
                       'MINIBATCH_SIZE': 1000,
                       'MEMORY_SIZE': 100_000,
                       'TARGET_SYNC_FREQ': 100,
                       'EPOCHS_PER_FIT': 1,

                       'GAMMA': 0.95,
                       'TAU': 1,
                       'EPSILON': 0.01,
                       'EPSILON_DECAY': 0.997,
                       'MIN_EPSILON': 0.01,
                       'DECAY_BEFORE_TRAINING': False,

                       'PATIENCE': 200,
                       'PATIENCE_MEMORY_SIZE': 100,
                       'STOP_EARLY': False, },
        '18x18': {'OLD_ACTIONS': True,
                  'HEIGHT': 18,
                  'WIDTH': 18,
                  'MIN_FOODS': 1,
                  'MAX_FOODS': 1,
                  'MIN_START_LENGTH': 4,
                  'MAX_START_LENGTH': 4,

                  'ROUNDS': 20,
                  'FITS': 1000,
                  'ENVS': 5,
                  'FIT_FREQ': 30,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 200_000,
                  'MIN_MEMORY_SIZE': 30_000,
                  'TARGET_SYNC_FREQ': 100,
                  'EPOCHS_PER_FIT': 1,

                  'GAMMA': 0.95,
                  'TAU': 1,
                  'EPSILON': 0.1,
                  'EPSILON_DECAY': 0.99,
                  'MIN_EPSILON': 0.01,
                  'DECAY_BEFORE_TRAINING': True,

                  'PATIENCE': 200,
                  'PATIENCE_MEMORY_SIZE': 100,
                  'STOP_EARLY': False, },
        '22x22': {'OLD_ACTIONS': True,
                  'HEIGHT': 22,
                  'WIDTH': 22,
                  'MIN_FOODS': 1,
                  'MAX_FOODS': 1,
                  'MIN_START_LENGTH': 4,
                  'MAX_START_LENGTH': 18,

                  'ROUNDS': 10,
                  'FITS': 1000,
                  'ENVS': 5,
                  'FIT_FREQ': 30,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 200_000,
                  'MIN_MEMORY_SIZE': 30_000,
                  'TARGET_SYNC_FREQ': 100,
                  'EPOCHS_PER_FIT': 1,

                  'GAMMA': 0.95,
                  'TAU': 1,
                  'EPSILON': 0.1,
                  'EPSILON_DECAY': 0.9998,
                  'MIN_EPSILON': 0.05,
                  'DECAY_BEFORE_TRAINING': True},
        '30x30 4': {'OLD_ACTIONS': False,
                    'HEIGHT': 30,
                    'WIDTH': 30,
                    'MIN_FOODS': 1,
                    'MAX_FOODS': 1,
                    'MIN_START_LENGTH': 4,
                    'MAX_START_LENGTH': 26,

                    'ROUNDS': 10,
                    'FITS': 1000,
                    'ENVS': 100,
                    'FIT_FREQ': 1,
                    'MINIBATCH_SIZE': 1000,
                    'MEMORY_SIZE': 200_000,
                    'MIN_MEMORY_SIZE': 10_000,
                    'TARGET_SYNC_FREQ': 100,
                    'EPOCHS_PER_FIT': 1,

                    'GAMMA': 0.95,
                    'TAU': 0.1,
                    'EPSILON': 0.05,
                    'EPSILON_DECAY': 1,
                    'MIN_EPSILON': 0.05,
                    'DECAY_BEFORE_TRAINING': True},
        '40x40': {'OLD_ACTIONS': True,
                  'HEIGHT': 40,
                  'WIDTH': 40,
                  'MIN_FOODS': 1,
                  'MAX_FOODS': 1,
                  'MIN_START_LENGTH': 4,
                  'MAX_START_LENGTH': 4,

                  'ROUNDS': 20,
                  'FITS': 1000,
                  'ENVS': 100,
                  'FIT_FREQ': 1,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 200_000,
                  'MIN_MEMORY_SIZE': 10_000,
                  'TARGET_SYNC_FREQ': 100,
                  'EPOCHS_PER_FIT': 1,

                  'GAMMA': 0.95,
                  'TAU': 1,
                  'EPSILON': 1,
                  'EPSILON_DECAY': 0.9998,
                  'MIN_EPSILON': 0.05,
                  'DECAY_BEFORE_TRAINING': True},
        '50x50': {'OLD_ACTIONS': True,
                  'HEIGHT': 50,
                  'WIDTH': 50,
                  'MIN_FOODS': 1,
                  'MAX_FOODS': 1,
                  'MIN_START_LENGTH': 4,
                  'MAX_START_LENGTH': 4,

                  'ROUNDS': 7,
                  'FITS': 1000,
                  'ENVS': 100,
                  'FIT_FREQ': 1,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 200_000,
                  'MIN_MEMORY_SIZE': 10_000,
                  'TARGET_SYNC_FREQ': 100,
                  'EPOCHS_PER_FIT': 1,

                  'GAMMA': 0.95,
                  'TAU': 1,
                  'EPSILON': 1,
                  'EPSILON_DECAY': 0.9995,
                  'MIN_EPSILON': 0.05,
                  'DECAY_BEFORE_TRAINING': True},
        '15x15 4': {'OLD_ACTIONS': False,
                    'HEIGHT': 15,
                    'WIDTH': 15,
                    'MIN_FOODS': 1,
                    'MAX_FOODS': 1,
                    'MIN_START_LENGTH': 4,
                    'MAX_START_LENGTH': 4,

                    'ROUNDS': 100,
                    'FITS': 1000,
                    'ENVS': 100,
                    'FIT_FREQ': 1,
                    'MINIBATCH_SIZE': 1000,
                    'MEMORY_SIZE': 1_000_000,
                    'MIN_MEMORY_SIZE': 100_000,
                    'TARGET_SYNC_FREQ': 100,
                    'EPOCHS_PER_FIT': 1,

                    'GAMMA': 0.95,
                    'TAU': 1,
                    'EPSILON': 1,
                    'EPSILON_DECAY': 0.999925,
                    'MIN_EPSILON': 0.01,
                    'DECAY_BEFORE_TRAINING': True},
    }

# loading of training parameters
OLD_ACTIONS: bool = parameters[MODEL_NAME]['OLD_ACTIONS']
HEIGHT: int = parameters[MODEL_NAME]['HEIGHT']
WIDTH: int = parameters[MODEL_NAME]['WIDTH']
MIN_FOODS: int = parameters[MODEL_NAME]['MIN_FOODS']
MAX_FOODS: int = parameters[MODEL_NAME]['MAX_FOODS']
MIN_START_LENGTH: int = parameters[MODEL_NAME]['MIN_START_LENGTH']
MAX_START_LENGTH: int = parameters[MODEL_NAME]['MAX_START_LENGTH']

ROUNDS: int = parameters[MODEL_NAME]['ROUNDS']
FITS: int = parameters[MODEL_NAME]['FITS']  # for every .fit, {WORKERS * ENVS * FIT_FREQ} steps are done
ENVS: int = parameters[MODEL_NAME]['ENVS']  # vectorized. amount of parallel envs
FIT_FREQ: int = parameters[MODEL_NAME]['FIT_FREQ']
MINIBATCH_SIZE: int = parameters[MODEL_NAME]['MINIBATCH_SIZE']
MEMORY_SIZE: int = parameters[MODEL_NAME]['MEMORY_SIZE']
MIN_MEMORY_SIZE: int = parameters[MODEL_NAME]['MIN_MEMORY_SIZE']
TARGET_SYNC_FREQ: int = parameters[MODEL_NAME]['TARGET_SYNC_FREQ']  # amount of .fits between updates
EPOCHS_PER_FIT: int = parameters[MODEL_NAME]['EPOCHS_PER_FIT']  # 1 is probably always the best choice

GAMMA: float = parameters[MODEL_NAME]['GAMMA']
TAU: float = parameters[MODEL_NAME]['TAU']  # 1 = hard update
epsilon: float = parameters[MODEL_NAME]['EPSILON']
EPSILON_DECAY: float = parameters[MODEL_NAME]['EPSILON_DECAY']
MIN_EPSILON: float = parameters[MODEL_NAME]['MIN_EPSILON']
DECAY_BEFORE_TRAINING: bool = parameters[MODEL_NAME]['DECAY_BEFORE_TRAINING']

# file management
SAVE_NAME = MODEL_NAME + '.keras'  # loads saved model if exists, else creates a new model
LOG_NAME = MODEL_NAME + ' log'  # overrides existing files with the same name

if os.path.exists(MODEL_NAME + ' metrics.pkl'):
    metrics = logs.read_log(MODEL_NAME + ' metrics')
else:
    metrics = [['score', []], ['total_reward', []], ['episode_length', []]]


def exploration() -> bool:
    """

    applies e-greedy policy
    """
    return random.random() < epsilon


def update_epsilon() -> None:
    global epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY


def interact(interaction_model, vec_env: VecSnake, steps: int) -> list[
    tuple[NDArray[float], int, int, NDArray[float], bool]]:
    # set infrastructure
    transitions = []
    current_states = vec_env.get_states()
    # training loop
    for step in range(steps):
        actions = np.argmax(interaction_model.predict(tensor_reshape(current_states, is_batch=True), verbose=0),
                            axis=-1)  # predict
        actions = [random.randint(0, 2) if exploration() else action for action in actions]  # explore
        new_states, rewards, dones = vec_env.step(actions)  # perform interaction
        transitions += [(current_states[i], actions[i], rewards[i], new_states[i], dones[i]) for i in
                        range(vec_env.size)]  # save transitions
        current_states = np.array(new_states)
        # reset done environments
        for i in range(vec_env.size):
            if dones[i]:
                metrics[0][-1].append(vec_env.scores(i))
                metrics[1][-1].append(vec_env.total_rewards(i))
                metrics[2][-1].append(vec_env.episode_lengths(i))
                current_states[i] = vec_env.reset_one(i, random.randint(MIN_START_LENGTH, MAX_START_LENGTH),
                                                      random.randint(MIN_FOODS, MAX_FOODS))
    return transitions


def train_agent() -> None:
    # set training infrastructure
    old_model = None
    if OLD_MODEL_NAME != '':
        old_model = tf.keras.models.load_model(OLD_MODEL_NAME + '.keras')
    agent = DQNAgent(SAVE_NAME, MEMORY_SIZE, (HEIGHT, WIDTH), old_model)
    interaction_model = tf.keras.models.clone_model(agent.model)
    interaction_model.set_weights(agent.model.get_weights())
    if OLD_ACTIONS:
        vec_env = VecSnake3((agent.model.input_shape[1], agent.model.input_shape[2], ENVS), render=False)
    else:
        vec_env = VecSnake((agent.model.input_shape[1], agent.model.input_shape[2], ENVS), render=False)
    vec_env.reset_all([random.randint(MIN_START_LENGTH, MAX_START_LENGTH) for i in range(ENVS)],
                      [random.randint(MIN_FOODS, MAX_FOODS) for i in range(ENVS)])
    # define data according to format. only used for human monitoring.
    if os.path.exists(LOG_NAME + '.pkl'):
        data = logs.read_log(LOG_NAME)
        pretraining_iterations = data[1][0][-1]
        data[0][-1] = f'{MODEL_NAME} performance over {pretraining_iterations + FITS * ROUNDS} training iterations'
    else:
        pretraining_iterations = 0
        if verbose: print(f'{now()} : starts evaluating round 0 / {ROUNDS}')
        data = [
            ['training iterations', 'mean score', f'{MODEL_NAME} performance over {FITS * ROUNDS} training iterations'],
            [[0], [score_evaluation(agent.model)], MODEL_NAME]]

    # gather initial experience
    if verbose: print(f'{now()} : starts gathering initial experience')
    while len(agent.memory) < MIN_MEMORY_SIZE:
        agent.update_memory(interact(interaction_model, vec_env, FIT_FREQ))  # append experience replay
        if DECAY_BEFORE_TRAINING:
            update_epsilon()
    if verbose:
        print(f'{now()} : finished gathering initial experience')
        print(f'{now()} : starts {FITS * ROUNDS} training iterations. evaluation every {FITS}')
    # main training loop
    if multi_processing:
        pool = mp.Pool(processes=1)
    for r in range(1, ROUNDS + 1):
        round_start = now()
        if verbose: print(f'{now()} : starts training round {r} / {ROUNDS}')
        for n in range(1, FITS + 1):
            if multi_processing:
                interaction = pool.starmap_async(interact, [(interaction_model, vec_env, FIT_FREQ)])
                agent.fit_minibatch(MINIBATCH_SIZE, EPOCHS_PER_FIT, GAMMA, TAU, TARGET_SYNC_FREQ)
                transitions = interaction.get()[0]  # wait for interactions to finish
                agent.update_memory(transitions)  # append experience replay
            else:
                agent.update_memory(interact(interaction_model, vec_env, FIT_FREQ))  # append experience replay
                agent.fit_minibatch(MINIBATCH_SIZE, EPOCHS_PER_FIT, GAMMA, TAU, TARGET_SYNC_FREQ)  # train
            interaction_model.set_weights(agent.model.get_weights())  # sync models
            update_epsilon()  # manage exploration/exploitation
        agent.save(f'round {r} ' + SAVE_NAME)
        if verbose: print(f'{now()} : starts evaluating round {r} / {ROUNDS}')
        data[1][0] += [r * FITS + pretraining_iterations]
        data[1][1] += [score_evaluation(agent, old_actions=OLD_ACTIONS)]
        # checkpoint
        if save_log: logs.log_data(data, LOG_NAME)
        if save_metrics: logs.log_data(metrics, MODEL_NAME + ' metrics')
        if verbose: print(f'{now()} : evaluation result: {data[1][1][-1]} estimated finish: {estimation}')
        if live_emails:
            graphs.new_graph(data, False, LOG_NAME)
            emails.send_message(f'round {r} {LOG_NAME} live update', f'estimated finish: {estimation}\n\n' + str(data),
                                LOG_NAME + '.png')
        estimation = add_time((ROUNDS - r) * runtime(round_start))

    agent.save(SAVE_NAME)
    if save_log: logs.log_data(data, LOG_NAME)
    if save_metrics: logs.log_data(metrics, MODEL_NAME + ' metrics')
    if summary_email:
        graphs.new_graph(data, False, LOG_NAME)
        graphs.plot_metrics(metrics, 1000, MODEL_NAME)
        emails.send_message(LOG_NAME + ' summary', str(data), LOG_NAME + '.png')
    elif save_graph:
        graphs.new_graph(data, False, LOG_NAME)
        graphs.plot_metrics(metrics, 1000, MODEL_NAME)
    if verbose: print(f'{now()} : done training')


def main():
    train_agent()


if __name__ == '__main__':
    start = now()
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.TIME)
        ps.print_stats()
        save_profile = open(MODEL_NAME + ' profile', 'w')
        save_profile.write(s.getvalue())
        save_profile.close()
        print(MODEL_NAME + ' profile saved')
    else:
        main()

    print(f'runtime was {runtime(start)}')
