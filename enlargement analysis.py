# own modules
import emails
import graphs
import logs
from agents import DQNAgent, tensor_reshape
from environments import VecSnake
from evaluation import score_evaluation
from mytime import now, add_time, runtime

import random
import numpy as np
from numpy.typing import NDArray
import multiprocessing as mp
import cProfile
import pstats
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

profile: bool = False
save_log: bool = True
save_graph: bool = True
verbose: bool = True
summary_email: bool = True

LOG_NAME = '22x22 enlargement log'  # None for no log
INITIAL_MODEL_NAME = '22x22.keras'
BOARDS = [25, 30, 35, 40, 45, 50]
pool_size = 3  # five is too much
# initial_data = [
#     ['training iterations', 'mean score', "analysis of 22x22 enlargements.\ninitial model's mean score: 31.947"],
#     [[0, 400, 800, 1200, 1600, 2000], [0.855, 2.675, 4.851, 11.263, 16.695, 21.349], '23x23'],
#     [[0, 400, 800, 1200, 1600, 2000], [0.655, 1.97, 3.815, 7.925, 13.031, 16.782], '24x24'],
#     [[0, 400, 800, 1200, 1600, 2000], [0.559, 1.947, 3.578, 6.743, 11.342, 16.827], '25x25'],
#     [[0, 400, 800, 1200, 1600, 2000], [0.441, 1.862, 4.483, 9.946, 15.864, 19.272], '26x26'],
#     [[0, 400, 800, 1200, 1600, 2000], [0.28, 1.148, 3.151, 7.161, 13.796, 17.126], '27x27'],
#     [[0, 400, 800, 1200, 1600, 2000], [0.306, 0.869, 2.127, 4.848, 10.809, 16.368], '28x28'],
#     [[0, 400, 800, 1200, 1600, 2000], [0.2, 0.628, 1.35, 3.578, 8.662, 13.477], '29x29'],
#     [[0, 400, 800, 1200, 1600, 2000], [0.178, 0.553, 1.366, 2.975, 9.055, 14.093], '30x30'],
#     [[0, 400, 800, 1200, 1600, 2000], [0.195, 0.501, 0.974, 2.676, 6.848, 11.553], '31x31']]
initial_data = None

# training parameters
ROUNDS: int = 5
FITS: int = 1000  # for every .fit, {ENVS * FIT_FREQ} steps are done
ENVS: int = 10  # vectorized. amount of parallel envs
FIT_FREQ: int = 1  # steps in every env per training iteration
MINIBATCH_SIZE: int = 128
MEMORY_SIZE: int = 30_000
MIN_MEMORY_SIZE = 2_000
TARGET_SYNC_FREQ: int = 100  # training iterations between target updates
TAU: float = 1  # 1 = hard update
EPOCHS_PER_FIT: int = 1  # 1 is probably always the best choice
GAMMA: float = 0.95

EPSILON: float = 0.75
epsilon = EPSILON
EPSILON_DECAY: float = 0.9995
MIN_EPSILON: float = 0.1
DECAY_BEFORE_TRAINING: bool = True

MIN_FOODS: int = 1
MAX_FOODS: int = 1
MIN_START_LENGTH: int = 4
MAX_START_LENGTH: int = 4


def exploration() -> bool:
    """

    applies e-greedy policy
    """
    return random.random() < epsilon


def update_epsilon(reset: bool = False) -> None:
    global epsilon
    if reset:
        epsilon = EPSILON
    elif epsilon > MIN_EPSILON:
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
                current_states[i] = vec_env.reset_one(i, random.randint(MIN_START_LENGTH, MAX_START_LENGTH),
                                                      random.randint(MIN_FOODS, MAX_FOODS))
    return transitions


def train_round(agent: DQNAgent, vec_env: VecSnake) -> tuple[DQNAgent, VecSnake]:
    # gather initial experience
    while len(agent.memory) < MIN_MEMORY_SIZE:
        agent.update_memory(interact(agent.model, vec_env, FIT_FREQ))
        if DECAY_BEFORE_TRAINING:
            update_epsilon()
    # main training loop
    for n in range(1, FITS + 1):
        agent.update_memory(interact(agent.model, vec_env, FIT_FREQ))
        agent.fit_minibatch(MINIBATCH_SIZE, EPOCHS_PER_FIT, GAMMA, TAU, TARGET_SYNC_FREQ)
        update_epsilon()
    return agent, vec_env


def train_enlargements(initial_model, log_name: str) -> None:
    # data variable is only for human visualization
    if verbose: print(f'{now()}: starts evaluating initial model')
    if initial_data is None:
        data = [['training iterations', 'mean score',
                 f"analysis of {initial_model.input_shape[1]}x{initial_model.input_shape[2]} enlargements."
                 f"\ninitial model's mean score: {score_evaluation(initial_model)}"]]  # define general labels
    else:
        data = initial_data
    pool = mp.Pool(processes=pool_size)
    while BOARDS:
        pool_start = now()
        active_boards = []
        while len(active_boards) < pool_size and BOARDS:
            active_boards.append(BOARDS.pop(0))
        if verbose: print(f'{now()} : starts analyzing {active_boards}')
        agents = []
        envs = []
        for board in active_boards:
            agents.append(DQNAgent(f'analysis {board}x{board}.keras', MEMORY_SIZE, (board, board), initial_model))
            envs.append(VecSnake((board, board, ENVS), render=False))
            envs[-1].reset_all([random.randint(MIN_START_LENGTH, MAX_START_LENGTH) for i in range(ENVS)],
                               [random.randint(MIN_FOODS, MAX_FOODS) for i in range(ENVS)])
        if verbose: print(f'{now()} : evaluating untrained models {active_boards}')
        mean_scores = pool.map(score_evaluation, agents)
        for i in range(len(active_boards)):  # create a line for each agent
            data.append([[0], [mean_scores[i]], f'{agents[i].model.input_shape[1]}x{agents[i].model.input_shape[2]}'])
        if verbose: print(f'{now()} : untrained evaluation results: {mean_scores}')
        for r in range(1, ROUNDS + 1):
            round_start = now()
            if verbose: print(f'{now()} : training round {r} / {ROUNDS} of {active_boards}')
            results = pool.starmap(train_round, zip(agents, envs))  # train all
            agents = [result[0] for result in results]
            envs = [result[1] for result in results]
            if verbose: print(f'{now()} : evaluating round {r} / {ROUNDS} of {active_boards}')
            mean_scores = pool.map(score_evaluation, agents)  # evaluate all
            for i in range(-1, -(len(active_boards) + 1), -1):  # reverse iteration to handle only active boards
                data[i][0] += [r * FITS]  # append xs of each line
                data[i][1] += [mean_scores[i]]  # append ys of each line
            round_estimation = add_time((ROUNDS - r) * runtime(round_start))
            if verbose:
                print(f'{now()} : evaluation results: {mean_scores}\ndata = {data}')
                if r == ROUNDS:
                    print(f'{now()} : pool {active_boards} finished!')
                else:
                    print(f'pool {active_boards} is estimated to finish at {round_estimation}')
        for agent in agents: agent.save()  # save all
        pool_estimation = add_time(runtime(pool_start) * len(BOARDS) / pool_size)
        if verbose and BOARDS: print(f'{now()} : whole analysis is estimated to finish at {pool_estimation}')
    if save_log: logs.log_data(data, log_name)
    if summary_email:
        graphs.new_graph(data, False, log_name)
        emails.send_message(log_name + ' summary', f'data in code format:\n{data}', log_name + '.png')
    elif save_graph:
        graphs.new_graph(data, False, log_name)
    if verbose: print(f'{now()} : analysis done')


def main():
    train_enlargements(tf.keras.models.load_model(INITIAL_MODEL_NAME), LOG_NAME)


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
        save_profile = open('enlargement analysis profile', 'w')
        save_profile.write(s.getvalue())
        save_profile.close()
        print('enlargement analysis profile saved')
    else:
        main()

    print(f'runtime was {runtime(start)}')
