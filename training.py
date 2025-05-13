# Own modules
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce tensorflow's verbose
import tensorflow as tf
import multiprocessing as mp
# For profiling
import cProfile
import pstats
import io

profile: bool = True
multi_processing: bool = False  # Causes retracing in TensorFlow when pygame is present
save_log: bool = True  # Evaluation mean scores
save_metrics: bool = True  # Training scores, episode lengths, and total rewards
save_graph: bool = True  # Happens anyway if summary email is active
verbose: bool = True
summary_email: bool = True
live_emails: bool = True

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
                  'EPOCHS_PER_FIT': 1,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 100_000,
                  'TARGET_SYNC_FREQ': 100,
                  'TAU': 1,
                  'GAMMA': 0.95,
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
                  'EPOCHS_PER_FIT': 1,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 100_000,
                  'TARGET_SYNC_FREQ': 100,
                  'TAU': 1,
                  'GAMMA': 0.95,
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
                       'EPOCHS_PER_FIT': 1,
                       'MINIBATCH_SIZE': 1000,
                       'MEMORY_SIZE': 100_000,
                       'TARGET_SYNC_FREQ': 100,
                       'TAU': 1,
                       'GAMMA': 0.95,
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
                  'EPOCHS_PER_FIT': 1,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 200_000,
                  'MIN_MEMORY_SIZE': 30_000,
                  'TARGET_SYNC_FREQ': 100,
                  'TAU': 1,
                  'GAMMA': 0.95,
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
                  'EPOCHS_PER_FIT': 1,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 200_000,
                  'MIN_MEMORY_SIZE': 30_000,
                  'TARGET_SYNC_FREQ': 100,
                  'TAU': 1,
                  'GAMMA': 0.95,
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
                    'EPOCHS_PER_FIT': 1,
                    'MINIBATCH_SIZE': 1000,
                    'MEMORY_SIZE': 200_000,
                    'MIN_MEMORY_SIZE': 10_000,
                    'TARGET_SYNC_FREQ': 100,
                    'TAU': 0.1,
                    'GAMMA': 0.95,
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
                  'EPOCHS_PER_FIT': 1,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 200_000,
                  'MIN_MEMORY_SIZE': 10_000,
                  'TARGET_SYNC_FREQ': 100,
                  'TAU': 1,
                  'GAMMA': 0.95,
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
                  'EPOCHS_PER_FIT': 1,
                  'MINIBATCH_SIZE': 1000,
                  'MEMORY_SIZE': 200_000,
                  'MIN_MEMORY_SIZE': 10_000,
                  'TARGET_SYNC_FREQ': 100,
                  'TAU': 1,
                  'GAMMA': 0.95,
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
                    'EPOCHS_PER_FIT': 1,
                    'MINIBATCH_SIZE': 1000,
                    'MEMORY_SIZE': 1_000_000,
                    'MIN_MEMORY_SIZE': 100_000,
                    'TARGET_SYNC_FREQ': 100,
                    'TAU': 1,
                    'GAMMA': 0.95,
                    'EPSILON': 1,
                    'EPSILON_DECAY': 0.999925,
                    'MIN_EPSILON': 0.01,
                    'DECAY_BEFORE_TRAINING': True},
        '17x17 demo': {'OLD_ACTIONS': False,
                       'HEIGHT': 17,
                       'WIDTH': 17,
                       'MIN_FOODS': 1,
                       'MAX_FOODS': 1,
                       'MIN_START_LENGTH': 4,
                       'MAX_START_LENGTH': 4,
                       'ROUNDS': 100,
                       'FITS': 1000,
                       'ENVS': 100,
                       'FIT_FREQ': 1,
                       'EPOCHS_PER_FIT': 1,
                       'MINIBATCH_SIZE': 1000,
                       'MEMORY_SIZE': 1_000_000,
                       'MIN_MEMORY_SIZE': 100_000,
                       'TARGET_SYNC_FREQ': 100,
                       'TAU': 1,
                       'GAMMA': 0.95,
                       'EPSILON': 1,
                       'EPSILON_DECAY': 0.999925,
                       'MIN_EPSILON': 0.01,
                       'DECAY_BEFORE_TRAINING': True},
        '16x16 profile demo': {'OLD_ACTIONS': False,
                               'HEIGHT': 16,
                               'WIDTH': 16,
                               'MIN_FOODS': 1,
                               'MAX_FOODS': 1,
                               'MIN_START_LENGTH': 4,
                               'MAX_START_LENGTH': 4,
                               'ROUNDS': 10,
                               'FITS': 500,
                               'ENVS': 10,
                               'FIT_FREQ': 1,
                               'EPOCHS_PER_FIT': 1,
                               'MINIBATCH_SIZE': 100,
                               'MEMORY_SIZE': 30_000,
                               'MIN_MEMORY_SIZE': 1_000,
                               'TARGET_SYNC_FREQ': 50,
                               'TAU': 1,
                               'GAMMA': 0.95,
                               'EPSILON': 1,
                               'EPSILON_DECAY': 0.999,
                               'MIN_EPSILON': 0.01,
                               'DECAY_BEFORE_TRAINING': True},
    }

# Loading of training parameters
OLD_ACTIONS: bool = parameters[MODEL_NAME]['OLD_ACTIONS']
HEIGHT: int = parameters[MODEL_NAME]['HEIGHT']
WIDTH: int = parameters[MODEL_NAME]['WIDTH']
MIN_FOODS: int = parameters[MODEL_NAME]['MIN_FOODS']
MAX_FOODS: int = parameters[MODEL_NAME]['MAX_FOODS']
MIN_START_LENGTH: int = parameters[MODEL_NAME]['MIN_START_LENGTH']
MAX_START_LENGTH: int = parameters[MODEL_NAME]['MAX_START_LENGTH']
ROUNDS: int = parameters[MODEL_NAME]['ROUNDS']
FITS: int = parameters[MODEL_NAME]['FITS']  # For every .fit, {ENVS * FIT_FREQ} steps are done
ENVS: int = parameters[MODEL_NAME]['ENVS']  # Vectorized. Amount of parallel envs
FIT_FREQ: int = parameters[MODEL_NAME]['FIT_FREQ']
EPOCHS_PER_FIT: int = parameters[MODEL_NAME]['EPOCHS_PER_FIT']  # 1 is probably always the best choice
MINIBATCH_SIZE: int = parameters[MODEL_NAME]['MINIBATCH_SIZE']
MEMORY_SIZE: int = parameters[MODEL_NAME]['MEMORY_SIZE']
MIN_MEMORY_SIZE: int = parameters[MODEL_NAME]['MIN_MEMORY_SIZE']
TARGET_SYNC_FREQ: int = parameters[MODEL_NAME]['TARGET_SYNC_FREQ']  # Amount of .fits between updates
TAU: float = parameters[MODEL_NAME]['TAU']  # 1 = hard update
GAMMA: float = parameters[MODEL_NAME]['GAMMA']
epsilon: float = parameters[MODEL_NAME]['EPSILON']
EPSILON_DECAY: float = parameters[MODEL_NAME]['EPSILON_DECAY']
MIN_EPSILON: float = parameters[MODEL_NAME]['MIN_EPSILON']
DECAY_BEFORE_TRAINING: bool = parameters[MODEL_NAME]['DECAY_BEFORE_TRAINING']

# File management
SAVE_NAME = MODEL_NAME + '.keras'
LOG_NAME = MODEL_NAME + ' log'

# Define metrics globally for usage in interact()
if os.path.exists(MODEL_NAME + ' metrics.pkl'):
    metrics = logs.read_log(MODEL_NAME + ' metrics')
else:
    metrics = [['score', []], ['total_reward', []], ['episode_length', []]]


def exploration() -> bool:
    """

    Applies epsilon-greedy policy
    :return: Whether to explore
    """
    return random.random() < epsilon


def update_epsilon() -> None:
    """
    Applies one exponential decay update to epsilon
    """
    global epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY


def interact(interaction_model, vec_env: VecSnake, steps: int) -> list[
    tuple[NDArray[float], int, int, NDArray[float], bool]]:
    """

    :param interaction_model: A keras neural network
    :param vec_env: A vectorized environment with attribute continuous=True
    :param steps: Amount of interaction steps
    :return: A list of experiences
    """
    experiences = []  # Initialize interaction storage data structure
    current_states = vec_env.get_states()  # Retrieve initial states
    # Interaction loop
    for step in range(steps):
        qs = interaction_model.predict(tensor_reshape(current_states, is_batch=True), verbose=0)  # Predict Q values
        actions = np.argmax(qs, axis=-1)  # Extract the actions with maximal Q values
        actions = [random.randint(0, 3) if exploration() else action for action in actions]  # Explore
        new_states, rewards, dones = vec_env.step(actions)  # Perform interaction
        experiences += [experience for experience in
                        zip(current_states, actions, rewards, new_states, dones, strict=True)]  # Store experiences
        current_states = np.array(new_states)  # Update states for the next step
        # Reset done environments
        for i in range(vec_env.size):
            if dones[i]:
                # Store game metrics
                metrics[0][-1].append(vec_env.scores(i))
                metrics[1][-1].append(vec_env.total_rewards(i))
                metrics[2][-1].append(vec_env.episode_lengths(i))
                # Reset
                current_states[i] = vec_env.reset_one(i, random.randint(MIN_START_LENGTH, MAX_START_LENGTH),
                                                      random.randint(MIN_FOODS, MAX_FOODS))
    return experiences


def train_agent() -> None:
    # ------------- Initialization Sequence -------------
    # Load a trained model for enlargement
    if OLD_MODEL_NAME != '':
        old_model = tf.keras.models.load_model(OLD_MODEL_NAME + '.keras')
    else:
        old_model = None
    # Initialize agent
    agent = DQNAgent(SAVE_NAME, MEMORY_SIZE, (HEIGHT, WIDTH), old_model)
    interaction_model = tf.keras.models.clone_model(agent.model)
    interaction_model.set_weights(agent.model.get_weights())
    # Initialize environment
    if OLD_ACTIONS:
        vec_env = VecSnake3((agent.model.input_shape[1], agent.model.input_shape[2], ENVS), render=False)
    else:
        vec_env = VecSnake((agent.model.input_shape[1], agent.model.input_shape[2], ENVS), render=False)
    vec_env.reset_all([random.randint(MIN_START_LENGTH, MAX_START_LENGTH) for i in range(ENVS)],
                      [random.randint(MIN_FOODS, MAX_FOODS) for i in range(ENVS)])
    # Define evaluation eval_data structure according to format. Used for human monitoring and logging.
    if os.path.exists(LOG_NAME + '.pkl'):  # Continue an existing log
        eval_data = logs.read_log(LOG_NAME)
        past_iterations = eval_data[1][0][-1]
        eval_data[0][-1] = f'{MODEL_NAME} evaluation score over {past_iterations + FITS * ROUNDS} training iterations'
    else:  # Create a new log
        past_iterations = 0
        if verbose: print(f'{now()} : starts evaluating round 0 / {ROUNDS}')
        eval_data = [['training iterations', 'mean score',
                      f'{MODEL_NAME} evaluation score over {FITS * ROUNDS} training iterations'],
                     [[0], [score_evaluation(agent.model)], MODEL_NAME]]
    # Gather initial experience
    if verbose: print(f'{now()} : starts gathering initial experience')
    while len(agent.memory) < MIN_MEMORY_SIZE:
        experiences = interact(interaction_model, vec_env, FIT_FREQ)  # Generate experiences
        agent.update_memory(experiences)  # Append experience buffer
        # Decay epsilon if needed
        if DECAY_BEFORE_TRAINING:
            update_epsilon()
    if verbose:
        print(f'{now()} : finished gathering initial experience')
        print(f'{now()} : starts {FITS * ROUNDS} training iterations. evaluation every {FITS}')
    # Initialize multi-processing if needed
    if multi_processing:
        pool = mp.Pool(processes=1)

    # ------------- Main Training Loop -------------
    for r in range(1, ROUNDS + 1):
        round_start = now()  # Save start time for runtime estimation calculation
        if verbose: print(f'{now()} : starts training round {r} / {ROUNDS}')
        # Training iterations
        for n in range(1, FITS + 1):
            if multi_processing:
                interaction = pool.starmap_async(interact, [
                    (interaction_model, vec_env, FIT_FREQ)])  # Start asynchronous interaction (data generation)
                agent.fit_minibatch(MINIBATCH_SIZE, EPOCHS_PER_FIT, GAMMA, TAU,
                                    TARGET_SYNC_FREQ)  # Train on previously collected experiences
                experiences = interaction.get()[0]  # Wait for interaction to finish
                agent.update_memory(experiences)  # Append experience buffer
            else:
                experiences = interact(interaction_model, vec_env, FIT_FREQ)  # Generate experiences
                agent.update_memory(experiences)  # Append experience buffer
                agent.fit_minibatch(MINIBATCH_SIZE, EPOCHS_PER_FIT, GAMMA, TAU, TARGET_SYNC_FREQ)  # Train
            interaction_model.set_weights(agent.model.get_weights())  # Sync models (important for multi-processing)
            update_epsilon()  # Decay epsilon to manage exploration/exploitation
        # Evaluate and store result
        if verbose: print(f'{now()} : starts evaluating round {r} / {ROUNDS}')
        eval_data[1][0] += [r * FITS + past_iterations]
        eval_data[1][1] += [score_evaluation(agent, old_actions=OLD_ACTIONS)]
        # Checkpoint
        agent.save(f'round {r} ' + SAVE_NAME)  # Save model
        if save_log: logs.log_data(eval_data, LOG_NAME)  # Save evaluation data
        if save_metrics: logs.log_data(metrics, MODEL_NAME + ' metrics')  # Save training metrics
        estimation = add_time((ROUNDS - r) * runtime(round_start))  # Estimate run finish time
        if verbose: print(f'{now()} : evaluation result: {eval_data[1][1][-1]} estimated finish: {estimation}')
        if live_emails:  # Send a training status email
            graphs.new_graph(eval_data, False, LOG_NAME)
            emails.send_message(f'round {r}/{ROUNDS} live update - {LOG_NAME}',
                                f'Estimated finish: {estimation}\nRaw evaluation eval_data so far:\n{eval_data}',
                                [LOG_NAME + '.png'])

    # ------------- Finalization Sequence -------------
    agent.save(SAVE_NAME)  # Save model
    if save_log: logs.log_data(eval_data, LOG_NAME)  # Save evaluation data
    if save_metrics: logs.log_data(metrics, MODEL_NAME + ' metrics')  # Save training metrics
    if summary_email:
        # Define email content
        subject = MODEL_NAME + ' training summary'
        body = (f'Finished training {MODEL_NAME}\n'
                f'Training started at {start} and finished at {now()} with a total runtime of {runtime(start)}\n'
                f'This email contains the following attachments:\n'
                f'- the trained model\n'
                f'- evaluation & training metrics\n'
                f'- graphs of the metrics')
        attachments = [SAVE_NAME, LOG_NAME + '.pkl', f'{MODEL_NAME} metrics.pkl', f'{MODEL_NAME} log.png',
                       f'{MODEL_NAME} score.png', f'{MODEL_NAME} episode_length.png', f'{MODEL_NAME} total_reward.png']
        # Smoothen evaluation data
        eval_data[1][2] = 'Raw'
        eval_data[1].append(0.3)
        moving_average_size = 5
        eval_data.append(
            [eval_data[1][0][moving_average_size - 1:], graphs.smoothen(eval_data[1][1], moving_average_size),
             f'Moving average ({moving_average_size})'])
        # Generate & save graphs
        graphs.new_graph(eval_data, False, LOG_NAME)
        graphs.plot_metrics(metrics, 1000, MODEL_NAME)
        # Send email
        emails.send_message(subject, body, attachments)
    elif save_graph:
        # Smoothen evaluation data
        eval_data[1][2] = 'Raw'
        eval_data[1].append(0.3)
        moving_average_size = 5
        eval_data.append(
            [eval_data[1][0][moving_average_size - 1:], graphs.smoothen(eval_data[1][1], moving_average_size),
             f'Moving average ({moving_average_size})'])
        # Generate & save graphs
        graphs.new_graph(eval_data, False, LOG_NAME)
        graphs.plot_metrics(metrics, 1000, MODEL_NAME)
    if verbose: print(f'{now()} : done training')


def main():
    train_agent()


if __name__ == '__main__':
    start = now()  # Save start time for runtime calculation
    if profile:
        # Profile using python's official documentation's code example
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.TIME)
        ps.print_stats()
        # Save
        with open(MODEL_NAME + ' profile', 'w') as save_file:
            save_file.write(s.getvalue())
            save_file.close()
        print(MODEL_NAME + ' profile saved')
    else:
        main()

    print(f'runtime was {runtime(start)}')  # Calculate and output total runtime
