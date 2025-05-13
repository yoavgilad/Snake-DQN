import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce tensorflow's verbose
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hide pygame's welcome output
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D
from environments import Snake
import numpy as np
from numpy.typing import NDArray
import random
from collections import deque
import pygame


def tensor_reshape(obs: NDArray[float], is_batch: bool = False) -> NDArray[float]:
    """

    Prepares observations for .predict
    :param obs: A single state or a batch of states
    :param is_batch: Whether a batch or a single state were given
    :return: Observations ready for .predict
    """
    if is_batch:
        return obs.reshape((obs.shape[0], obs.shape[1], obs.shape[2], 1))  # vectorized reshape
    return obs.reshape((1, obs.shape[0], obs.shape[1], 1))  # single state reshape


def unpack_tensor_reshape(obs: NDArray[float], is_batch: bool = False) -> NDArray[float]:
    """

    Undoes the function tensor_reshape
    :param obs: A single state or a batch of states
    :param is_batch: Whether a batch or a single state were given
    :return: Observations in their regular form
    """
    if is_batch:
        return obs.reshape((obs.shape[0], obs.shape[1], obs.shape[2]))  # vectorized reshape
    return obs.reshape((obs.shape[1], obs.shape[2]))  # single state reshape


def mask_loss(state: NDArray[float], body: list, qs: NDArray[float] | list[float]):
    # Prepare variables
    head_i, head_j = body[-1]
    qs = qs[0]
    old_action = np.argmax(qs)
    mask = min(qs) - 1
    # It's impossible to move backwards, and trying so results in forward movement.
    # Therefore, we mask every direction with an obstacle, unless the snake came from it:
    # Up
    if state[head_i - 1, head_j] == Snake.BODY or state[head_i - 1, head_j] == Snake.WALL and head_i - 1 != body[-2][0]:
        qs[0] = mask
    # Right
    if state[head_i, head_j + 1] == Snake.BODY or state[head_i, head_j + 1] == Snake.WALL and head_j + 1 != body[-2][1]:
        qs[1] = mask
    # Down
    if state[head_i + 1, head_j] == Snake.BODY or state[head_i + 1, head_j] == Snake.WALL and head_i + 1 != body[-2][0]:
        qs[2] = mask
    # Left
    if state[head_i, head_j - 1] == Snake.BODY or state[head_i, head_j - 1] == Snake.WALL and head_j - 1 != body[-2][1]:
        qs[3] = mask
    # And if there's an obstacle straight ahead, mask the backwards direction as well:
    if head_i - 1 == body[-2][0] and qs[2] == mask: qs[0] = mask
    if head_j + 1 == body[-2][1] and qs[3] == mask: qs[1] = mask
    if head_i + 1 == body[-2][0] and qs[0] == mask: qs[2] = mask
    if head_j - 1 == body[-2][1] and qs[1] == mask: qs[3] = mask
    # Notify user if masking changed the chosen action
    if np.argmax(qs) != old_action:
        print('used mask')
    return qs


class DQNAgent:
    def __init__(self, save_name: str, memory_size: int, shape: tuple[int, int], old_model=None):
        # Creation of a new model if it doesn't exist already
        if not os.path.exists(save_name):
            self.create_model(shape, old_model).save(save_name)
        self.save_name = save_name
        # Load model
        self.model = tf.keras.models.load_model(save_name)
        # Initialize target network
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.target_sync_counter = 0
        # Initialize experience buffer
        self.memory = deque(maxlen=memory_size)

    def create_model(self, new_shape: tuple[int, int], old_model=None, action_space: int = 4):
        """

        :param action_space: Output shape; either 3 or 4
        :param new_shape: New model's input shape
        :param old_model: If passed, weights will be transferred.
        :return: A new model
        """

        # New model definition. Should be identical to old model except input layer.
        inputs = Input(shape=(new_shape[0], new_shape[1], 1))
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(inputs)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        value = Dense(1, activation='linear')(x)
        advantage = Dense(action_space, activation='linear')(x)
        mean_advantage = tf.keras.ops.mean(advantage, axis=1, keepdims=True)
        q_values = value + (advantage - mean_advantage)

        new_model = tf.keras.Model(inputs=inputs, outputs=q_values)

        # Transfer weights if given
        if old_model is not None:
            for old_layer, new_layer in zip(old_model.layers, new_model.layers):
                if isinstance(new_layer, Dense):
                    # Zero padding to the end of the array.
                    old_weights, old_biases = old_layer.get_weights()
                    pad_size = new_layer.get_weights()[0].shape[0] - old_weights.shape[0]
                    padded_weights = np.pad(old_weights, ((0, pad_size), (0, 0)),
                                            mode='constant', constant_values=0)
                    new_layer.set_weights([padded_weights, old_biases])
                elif isinstance(new_layer, Conv2D):
                    # No padding
                    new_layer.set_weights(old_layer.get_weights())

        new_model.compile(loss='mean_squared_error', optimizer='adam')
        return new_model

    def update_memory(self, experiences: list[tuple]) -> None:
        self.memory.extend(experiences)

    def fit_minibatch(self, minibatch_size: int, epochs: int, gamma: float, tau: float, target_sync_freq: int) -> None:
        minibatch = random.sample(self.memory, minibatch_size)  # Retrieve experiences
        # Predict Q(s, a)
        current_states = tensor_reshape(np.array([experience[0] for experience in minibatch]), is_batch=True)
        current_qs_minibatch = self.model.predict(current_states, verbose=0)
        # Predict Q(s', a')
        new_states = tensor_reshape(np.array([experience[3] for experience in minibatch]), is_batch=True)
        future_qs_minibatch = self.target_model.predict(new_states, verbose=0)
        # Initialize data structures
        x = []
        y = []
        # Label data
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            # Calculate Q(s, a) using Bellman equation
            if done:
                new_q = reward
            else:
                new_q = reward + gamma * np.max(future_qs_minibatch[index])
            # Combine prediction & calculation to create y labels
            current_qs = current_qs_minibatch[index]
            current_qs[action] = new_q
            # Append to training dataset
            x.append(current_state)
            y.append(current_qs)
        # Train the model
        self.model.fit(np.array(x), np.array(y), epochs=epochs, batch_size=minibatch_size, verbose=0, shuffle=False)
        # Synchronize target network if needed
        self.target_sync_counter += 1
        if self.target_sync_counter >= target_sync_freq:
            model_weights = np.array(self.model.get_weights(), dtype=object)
            target_weights = np.array(self.target_model.get_weights(), dtype=object)
            new_weights = tau * model_weights + (1.0 - tau) * target_weights
            self.target_model.set_weights(new_weights)
            self.target_sync_counter = 0

    def save(self, save_name: str = None):
        if save_name is None:
            self.model.save(self.save_name)
        else:
            self.model.save(save_name)


class HumanPlayer:
    def __init__(self, input_shape: tuple[int, int, int, int]):
        self.input_shape = input_shape
        self.action = 0

    def predict(self, state=None, verbose=None) -> list[NDArray[float]]:
        # Receive human input
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == pygame.K_UP:
                    self.action = 0
                if key == pygame.K_RIGHT:
                    self.action = 1
                if key == pygame.K_DOWN:
                    self.action = 2
                if key == pygame.K_LEFT:
                    self.action = 3
        # Imitate keras output structure
        q_values = np.zeros(4)
        q_values[self.action] = 1
        return [q_values]


class RandomPlayer:
    def __init__(self, input_shape: tuple[int, int, int, int]):
        self.input_shape = input_shape

    def predict(self, current_state=None, verbose=None) -> list[NDArray[float]]:
        # Imitate keras output structure with random values, for each parallel game
        q_values = [np.array([random.random() for action in range(4)]) for env in range(self.input_shape[0])]
        return q_values
