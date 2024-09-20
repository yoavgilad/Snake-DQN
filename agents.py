import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D
from environments import Snake
import numpy as np
from numpy.typing import NDArray
import random
from collections import deque
import pygame


def tensor_reshape(obs: NDArray[float], is_batch: bool = False) -> NDArray[float]:  # for tensorflow
    """

    prepares observations for .predict
    :param obs: a single state or a batch of states
    :param is_batch:
    :return: observations ready for .predict
    """
    if is_batch:
        return obs.reshape((obs.shape[0], obs.shape[1], obs.shape[2], 1))  # vectorized observation reshape
    return obs.reshape((1, obs.shape[0], obs.shape[1], 1))  # single observation reshape


def unpack_tensor_reshape(obs: NDArray[float], is_batch: bool = False) -> NDArray[float]:
    if is_batch:
        return obs.reshape((obs.shape[0], obs.shape[1], obs.shape[2]))
    return obs.reshape((obs.shape[1], obs.shape[2]))


def mask_loss(state: NDArray[float], body: list, qs: NDArray[float] | list[float]):
    for i, x in enumerate(state.tolist()):
        if Snake.HEAD in x:
            head_i, head_j = i, x.index(Snake.HEAD)
            break
    qs = qs[0]
    old_action = np.argmax(qs)
    mask = min(qs) - 1
    # up
    if (state[head_i - 1, head_j] == Snake.BODY or state[head_i - 1, head_j] == Snake.WALL) and (
            head_i - 1 != body[-2][0] or head_j != body[-2][1]): qs[0] = mask
    # right
    if (state[head_i, head_j + 1] == Snake.BODY or state[head_i, head_j + 1] == Snake.WALL) and (
            head_i != body[-2][0] or head_j + 1 != body[-2][1]): qs[1] = mask
    # down
    if (state[head_i + 1, head_j] == Snake.BODY or state[head_i + 1, head_j] == Snake.WALL) and (
            head_i + 1 != body[-2][0] or head_j != body[-2][1]): qs[2] = mask
    # left
    if (state[head_i, head_j - 1] == Snake.BODY or state[head_i, head_j - 1] == Snake.WALL) and (
            head_i != body[-2][0] or head_j - 1 != body[-2][1]): qs[3] = mask

    if head_i - 1 == body[-2][0] and head_j == body[-2][1] and qs[2] == mask: qs[0] = mask
    if head_i == body[-2][0] and head_j + 1 == body[-2][1] and qs[3] == mask: qs[1] = mask
    if head_i + 1 == body[-2][0] and head_j == body[-2][1] and qs[0] == mask: qs[2] = mask
    if head_i == body[-2][0] and head_j - 1 == body[-2][1] and qs[1] == mask: qs[3] = mask
    if np.argmax(qs) != old_action:
        print('used mask')
    return qs


class DQNAgent:

    def __init__(self, save_name: str, memory_size: int, shape: tuple[int, int], old_model=None):
        # creation of new models if they don't exist already
        if not os.path.exists(save_name):
            self.create_model(shape, old_model).save(save_name)
        self.save_name = save_name
        # load model
        self.model = tf.keras.models.load_model(save_name)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.target_sync_counter = 0
        self.memory = deque(maxlen=memory_size)

    def create_model(self, new_shape: tuple[int, int], old_model=None, action_space: int = 4):
        """

        :param action_space:
        :param new_shape: new model's input shape
        :param old_model: if passed, weights will be transferred.
        :return:
        """

        # new model definition. should be identical to old model except input layer.
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

        if old_model is not None:
            # transfer weights
            for old_layer, new_layer in zip(old_model.layers, new_model.layers):
                if isinstance(new_layer, Dense):
                    # zero padding. currently only at the start of the array. consider changing
                    old_weights, old_biases = old_layer.get_weights()
                    pad_size = new_layer.get_weights()[0].shape[0] - old_weights.shape[0]
                    padded_weights = np.pad(old_weights, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
                    new_layer.set_weights([padded_weights, old_biases])
                elif isinstance(new_layer, Conv2D):
                    # no padding
                    new_layer.set_weights(old_layer.get_weights())

        new_model.compile(loss='mean_squared_error', optimizer='adam')
        return new_model

    def update_memory(self, transitions: list[tuple]) -> None:
        self.memory.extend(transitions)

    def fit_minibatch(self, minibatch_size: int, epochs: int, gamma: float, tau: float, target_sync_freq: int) -> None:
        minibatch = random.sample(self.memory, minibatch_size)

        current_states = tensor_reshape(np.array([transition[0] for transition in minibatch]), is_batch=True)
        current_qs_minibatch = self.model.predict(current_states, verbose=0)

        new_states = tensor_reshape(np.array([transition[3] for transition in minibatch]), is_batch=True)
        future_qs_minibatch = self.target_model.predict(new_states, verbose=0)

        x = []
        y = []
        # create labeled data
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if done:
                new_q = reward
            else:
                new_q = reward + gamma * np.max(future_qs_minibatch[index])  # bellman equation
            # update model's predicted qs with label
            current_qs = current_qs_minibatch[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), epochs=epochs, batch_size=minibatch_size, verbose=0, shuffle=False)

        # sync target network if needed
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
        q_values = np.zeros(4)
        q_values[self.action] = 1
        return [q_values]


class RandomPlayer:
    def __init__(self, input_shape: tuple[int, int, int, int]):
        self.input_shape = input_shape

    def predict(self, current_state=None, verbose=None):
        q_values = [np.zeros(4) for i in range(self.input_shape[0])]
        for qs in q_values:
            qs[random.randint(0, 3)] = 1
        return q_values
