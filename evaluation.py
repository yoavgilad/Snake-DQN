import graphs
import logs
import numpy as np
import random
from environments import VecSnake, Snake, Snake3, VecSnake3
from agents import tensor_reshape, DQNAgent, HumanPlayer, mask_loss
from time import sleep
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame


def visual_evaluation(model, episodes: int = -1, loop_threshold: int = 200, foods: tuple[int, int] = (1, 1),
                      start_lengths: tuple[int, int] = (4, 4), fps: int = 15, save_best: bool = True,
                      old_actions: bool = False, mask: bool = False) -> float:
    """

    :param model: Either a Q-network or an agent
    :param episodes: -1 for endless evaluation
    :param loop_threshold: Negative value for no loop detection
    :param foods: (min foods, max foods). A random int in this range is chosen each episode
    :param start_lengths: (min start length, max start length). A random int in this range will be chosen each episode
    :param fps: FPS limit
    :param save_best: Whether to save the best game
    :param old_actions: True = relative movement, False = absolute movement
    :param mask: Whether to mask instantly-losing actions
    :return: Mean score of the evaluation episodes
    """
    # Extract network from an agent if given
    if isinstance(model, DQNAgent):
        model = model.model
    # Construct the relevant class for the action space
    if old_actions:
        env = Snake3((model.input_shape[1], model.input_shape[2]),
                     loop_threshold if not isinstance(model, HumanPlayer) else -1)
    else:
        env = Snake((model.input_shape[1], model.input_shape[2]),
                    loop_threshold if not isinstance(model, HumanPlayer) else -1)
    env.fps = fps  # Set fps
    # Initialize score tracking infrastructure
    scores = []
    max_score = 0
    # Main game loops
    for episode in range(1, episodes + 1):
        # Reset env and obtain the initial state
        current_state = env.reset(random.randint(start_lengths[0], start_lengths[1]),
                                  random.randint(foods[0], foods[1]))
        done = False
        if env.is_render: sleep(1)  # Wait before starting gameplay
        if save_best: game_log = [current_state]  # Initialize game recording
        # Game loop
        while not done:
            # Determine action
            qs = model.predict(tensor_reshape(current_state), verbose=0)
            if mask: qs = mask_loss(current_state, env.body, qs)
            action = np.argmax(qs, axis=-1)
            # Clean user input
            if env.is_render: pygame.event.get()
            new_state, reward, done = env.step(action)  # Perform action
            current_state = new_state  # Update state
            if save_best: game_log.append(current_state)  # Append game recording
        scores.append(env.score)  # Save score
        print(f'Game {episode}, Score: {scores[-1]}')
        if save_best:
            game_log.insert(0, env.score)  # Add score to the record as metadata
            # Save recording if it's a new high score and wasn't done by a human
            if not isinstance(model, HumanPlayer) and game_log[0] > max_score:
                max_score = game_log[0]
                if not os.path.exists(f'{env.HEIGHT}x{env.WIDTH} best game.pkl') or \
                        logs.read_log(f'{env.HEIGHT}x{env.WIDTH} best game')[0] < max_score:
                    logs.log_data(game_log, f'{env.HEIGHT}x{env.WIDTH} best game')
                    print('saved!')
        if env.is_render: sleep(1)  # Wait on final state

    return np.mean(np.array(scores))


def score_evaluation(model, episodes: int = 100, loop_threshold: int = 200, foods: tuple[int, int] = (1, 1),
                     start_lengths: tuple[int, int] = (4, 4), save_best: bool = False,
                     graph_save_name: str | None = None, old_actions: bool = False) -> float:
    """

    :param model: Either a Q-network or an agent
    :param episodes: All episodes will run in parallel
    :param loop_threshold: Negative value for no loop detection
    :param foods: (min foods, max foods). A random int in this range will be chosen each episode
    :param start_lengths: (min start length, max start length). A random int in this range is chosen each episode
    :param save_best: Whether to save a recording of the best game
    :param graph_save_name: Score distribution graph won't be saved if None
    :param old_actions: True = relative movement, False = absolute movement
    :return: Mean score of the evaluation episodes
    """
    # Extract network from an agent if given
    if isinstance(model, DQNAgent):
        model = model.model
    # Construct the relevant class for the action space
    if old_actions:
        vec_env = VecSnake3((model.input_shape[1], model.input_shape[2], episodes), loop_threshold, False, False)
    else:
        vec_env = VecSnake((model.input_shape[1], model.input_shape[2], episodes), loop_threshold, False, False)
    # Reset env and obtain the initial states
    current_states = vec_env.reset_all([random.randint(start_lengths[0], start_lengths[1]) for i in range(episodes)],
                                       [random.randint(foods[0], foods[1]) for i in range(episodes)])
    dones = [None]  # Initialize with some value to enter the main loop
    # Initialize recording if needed
    if save_best:
        game_logs = [[state] for state in current_states]
        best_log = [0]
    # Main loop of all parallel games
    while dones:
        # Determine and perform actions
        actions = np.argmax(model.predict(tensor_reshape(current_states, is_batch=True), verbose=0), axis=-1)
        new_states, rewards, dones = vec_env.step(actions)
        # Remove done environments
        for i in range(len(dones) - 1, -1, -1):
            if dones[i]:
                # Keep recording if it's a new high score, else discard it
                if save_best:
                    if vec_env.active_envs[i].score > best_log[0]:
                        best_log = [vec_env.active_envs[i].score] + game_logs.pop(i)
                    else:
                        game_logs.pop(i)
                dones.pop(i)
                new_states.pop(i)
                vec_env.close_one(i)
                vec_env.active_envs.pop(i)

        current_states = np.array(new_states)  # Update states
        if save_best:  # Append recordings
            for i in range(len(game_logs)): game_logs[i].append(current_states[i])
    # Save recording of the best game if it's a new high score
    if save_best:
        if not os.path.exists(f'{model.input_shape[1]}x{model.input_shape[2]} best game.pkl') or \
                logs.read_log(f'{model.input_shape[1]}x{model.input_shape[2]} best game')[0] < best_log[0]:
            logs.log_data(best_log, f'{model.input_shape[1]}x{model.input_shape[2]} best game')
            print(f'saved a game with a score of {best_log[0]}!')
    # Save score distribution
    if graph_save_name is not None:
        scores_dist = np.zeros(max(vec_env.scores()) + 1).astype(int)
        for score in vec_env.scores():
            scores_dist[score] += 1
        # Smoothen data
        moving_average = 10
        smooth_scores_dist = np.convolve(scores_dist, np.ones(moving_average) / moving_average, mode='valid')
        # Define graph configuration according to the format
        graph_data = [['score', 'number of games',
                       f'score distribution of {episodes} evaluation episodes ({model.input_shape[1]}x{model.input_shape[2]})'],
                      [[x for x in range(len(scores_dist))], scores_dist, 'Raw', 0.2],
                      [[x for x in range(moving_average // 2, len(smooth_scores_dist) + moving_average // 2)],
                       smooth_scores_dist, f'Moving average ({moving_average})']]
        # Generate and save graph
        graphs.new_graph(graph_data, False, graph_save_name)

    return np.mean(np.array(vec_env.scores()))
