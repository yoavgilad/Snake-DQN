import random
import numpy as np
from numpy.typing import NDArray
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame


class Snake:
    fps = 60
    rewards = {'food': 15, 'hit wall': -10, 'hit body': -10, 'survive': 0, 'move closer': 1, 'move further': -1}
    # move closer and move further rewards only apply if MAX_FOOD=1
    # representations for different tile types
    HEAD = 0.5
    BODY = -0.5
    FOOD = 1
    WALL = -1
    EMPTY = 0

    def __init__(self, shape: tuple[int, int], loop_threshold: int = -1, render: bool = True):
        """
        :param loop_threshold: negative value = no loop declaration
        """
        # declaring variables for comfortability
        self.HEIGHT, self.WIDTH = shape
        self.state: NDArray[float] = np.array([])
        self.body: list[tuple[int, int]] = []
        self.foods: list[tuple[int, int]] = []
        # metrics
        self.score: int = 0
        self.total_reward: int = 0
        self.episode_length: int = 0

        self.loop_threshold: int = loop_threshold  # consecutive moves without food before declaring loop
        self.loop_counter: int = 0
        self.is_render: bool = render
        if self.is_render:
            self.clock = pygame.time.Clock()
            self.WINDOW_SIZE = 750 // self.HEIGHT
            pygame.init()
            self.screen = pygame.display.set_mode((self.HEIGHT * self.WINDOW_SIZE, self.WIDTH * self.WINDOW_SIZE))
            pygame.display.set_caption('Snake')

    def render(self):
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                rect = pygame.Rect(x * self.WINDOW_SIZE, y * self.WINDOW_SIZE, self.WINDOW_SIZE, self.WINDOW_SIZE)
                match self.state[y, x]:
                    case self.EMPTY:
                        pygame.draw.rect(self.screen, (0, 150, 0), rect)
                        pygame.draw.rect(self.screen, (0, 175, 0), rect, 1)
                    case self.WALL:
                        pygame.draw.rect(self.screen, (128, 128, 128), rect)
                    case self.BODY:
                        pygame.draw.rect(self.screen, (0, 0, 125), rect)
                        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                    case self.HEAD:
                        pygame.draw.rect(self.screen, (0, 0, 255), rect)
                        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                    case self.FOOD:
                        pygame.draw.rect(self.screen, (255, 0, 0), rect)
                        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        pygame.display.flip()  # Update the display

    def reset(self, start_length: int = 4, foods: int = 1) -> NDArray[float]:
        # reset variables
        self.loop_counter = 0
        self.body = []
        self.foods = []
        self.score = 0
        self.total_reward = 0
        self.episode_length = 0
        self.state = np.zeros(shape=(self.HEIGHT, self.WIDTH))
        self.state[0] += self.WALL
        self.state[-1] += self.WALL
        for i in range(1, self.HEIGHT - 1):
            self.state[i][0] = self.WALL
            self.state[i][-1] = self.WALL

        # choose a random starting position
        start_pos = None
        look_dir = random.randint(1, 4)
        match look_dir:
            case 1:  # up
                start_pos = (random.randint(2, self.HEIGHT - start_length - 2),
                             random.randint(2, self.WIDTH - 3))
                for i in range(1, start_length):
                    self.state[start_pos[0] + i, start_pos[1]] = self.BODY
                    self.body.insert(0, (start_pos[0] + i, start_pos[1]))
            case 2:  # right
                start_pos = (random.randint(2, self.HEIGHT - 3),
                             random.randint(start_length + 1, self.WIDTH - 2))
                for i in range(1, start_length):
                    self.state[start_pos[0], start_pos[1] - i] = self.BODY
                    self.body.insert(0, (start_pos[0], start_pos[1] - i))
            case 3:  # down
                start_pos = (random.randint(start_length + 1, self.HEIGHT - 2),
                             random.randint(2, self.WIDTH - 3))
                for i in range(1, start_length):
                    self.state[start_pos[0] - i, start_pos[1]] = self.BODY
                    self.body.insert(0, (start_pos[0] - i, start_pos[1]))
            case 4:  # left
                start_pos = (random.randint(2, self.HEIGHT - 3),
                             random.randint(2, self.WIDTH - start_length - 2))
                for i in range(1, start_length):
                    self.state[start_pos[0], start_pos[1] + i] = self.BODY
                    self.body.insert(0, (start_pos[0], start_pos[1] + i))

        # add head
        self.body.append(start_pos)
        self.state[start_pos[0], start_pos[1]] = self.HEAD

        self.generate_food(foods)
        if self.is_render: self.render()
        return np.copy(self.state)

    def step(self, action: int) -> tuple[NDArray[float], int, bool]:
        reward = self.rewards['survive']
        done = False
        match action:
            case 0:  # up
                if self.body[-1][0] - 1 != self.body[-2][0]:
                    self.body.append((self.body[-1][0] - 1, self.body[-1][1]))
                else:
                    self.body.append((self.body[-1][0] + 1, self.body[-1][1]))
            case 1:  # right
                if self.body[-1][1] + 1 != self.body[-2][1]:
                    self.body.append((self.body[-1][0], self.body[-1][1] + 1))
                else:
                    self.body.append((self.body[-1][0], self.body[-1][1] - 1))
            case 2:  # down
                if self.body[-1][0] + 1 != self.body[-2][0]:
                    self.body.append((self.body[-1][0] + 1, self.body[-1][1]))
                else:
                    self.body.append((self.body[-1][0] - 1, self.body[-1][1]))
            case 3:  # left
                if self.body[-1][1] - 1 != self.body[-2][1]:
                    self.body.append((self.body[-1][0], self.body[-1][1] - 1))
                else:
                    self.body.append((self.body[-1][0], self.body[-1][1] + 1))

        match self.state[self.body[-1][0], self.body[-1][1]]:
            case self.EMPTY:
                if len(self.foods) == 1:
                    diff = (np.sqrt(
                        (self.body[-2][0] - self.foods[0][0]) ** 2 + (self.body[-2][1] - self.foods[0][1]) ** 2) -
                            np.sqrt((self.body[-1][0] - self.foods[0][0]) ** 2 + (
                                    self.body[-1][1] - self.foods[0][1]) ** 2))
                    if diff > 0:
                        reward = self.rewards['move closer']
                    elif diff < 0:
                        reward = self.rewards['move further']
                    else:
                        reward = self.rewards['survive']
                else:
                    reward = self.rewards['survive']
                self.state[self.body[-1][0], self.body[-1][1]] = self.HEAD
                self.state[self.body[-2][0], self.body[-2][1]] = self.BODY
                self.state[self.body[0][0], self.body[0][1]] = self.EMPTY
                self.body.pop(0)
                self.loop_counter += 1
                if self.loop_counter == self.loop_threshold:
                    done = True
            case self.FOOD:
                self.loop_counter = 0
                reward = self.rewards['food']
                self.state[self.body[-1][0], self.body[-1][1]] = self.HEAD
                self.state[self.body[-2][0], self.body[-2][1]] = self.BODY
                self.foods.remove(self.body[-1])
                self.generate_food(1)
                self.score += 1
            case self.BODY:
                reward = self.rewards['hit body']
                done = True
            case self.WALL:
                reward = self.rewards['hit wall']
                done = True
        self.episode_length += 1
        self.total_reward += reward
        if self.is_render:
            self.render()
            self.clock.tick(self.fps)
        return np.copy(self.state), reward, done

    def generate_food(self, num_foods: int = 1) -> None:
        for i in range(num_foods):
            pos = random.randint(0, self.HEIGHT - 1), random.randint(0, self.WIDTH - 1)
            while self.state[pos[0], pos[1]] != 0:
                pos = random.randint(0, self.HEIGHT - 1), random.randint(0, self.WIDTH - 1)
            self.state[pos[0], pos[1]] = self.FOOD
            self.foods.append(pos)

    def close(self):
        if self.is_render:
            pygame.quit()


class Snake3:
    fps = 60
    rewards = {'food': 15, 'hit wall': -10, 'hit body': -10, 'survive': 0, 'move closer': 1, 'move further': -1}
    # move closer and move further rewards only apply if MAX_FOOD=1
    # representations for different tile types
    HEAD = 0.5
    BODY = -0.5
    FOOD = 1
    WALL = -1
    EMPTY = 0

    def __init__(self, shape: tuple[int, int], loop_threshold: int = -1, render: bool = True):
        """
        :param loop_threshold: negative value = no loop declaration
        """
        # declaring variables for comfortability
        self.HEIGHT, self.WIDTH = shape
        self.state: NDArray[float] = np.array([])
        self.body: list[tuple[int, int]] = []
        self.foods: list[tuple[int, int]] = []
        # metrics
        self.score: int = 0
        self.total_reward: int = 0
        self.episode_length: int = 0

        self.loop_threshold: int = loop_threshold  # consecutive moves without food before declaring loop
        self.loop_counter: int = 0
        self.is_render: bool = render
        if self.is_render:
            self.clock = pygame.time.Clock()
            self.WINDOW_SIZE = 750 // self.HEIGHT
            pygame.init()
            self.screen = pygame.display.set_mode((self.HEIGHT * self.WINDOW_SIZE, self.WIDTH * self.WINDOW_SIZE))
            pygame.display.set_caption('Snake')

    def render(self):
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                rect = pygame.Rect(x * self.WINDOW_SIZE, y * self.WINDOW_SIZE, self.WINDOW_SIZE, self.WINDOW_SIZE)
                match self.state[y, x]:
                    case self.EMPTY:
                        pygame.draw.rect(self.screen, (0, 150, 0), rect)
                        pygame.draw.rect(self.screen, (0, 175, 0), rect, 1)
                    case self.WALL:
                        pygame.draw.rect(self.screen, (128, 128, 128), rect)
                    case self.BODY:
                        pygame.draw.rect(self.screen, (0, 0, 125), rect)
                        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                    case self.HEAD:
                        pygame.draw.rect(self.screen, (0, 0, 255), rect)
                        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                    case self.FOOD:
                        pygame.draw.rect(self.screen, (255, 0, 0), rect)
                        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        pygame.display.flip()  # Update the display

    def reset(self, start_length: int = 4, foods: int = 1) -> NDArray[float]:
        # reset variables
        self.loop_counter = 0
        self.body = []
        self.foods = []
        self.score = 0
        self.total_reward = 0
        self.episode_length = 0
        self.state = np.zeros(shape=(self.HEIGHT, self.WIDTH))
        self.state[0] += self.WALL
        self.state[-1] += self.WALL
        for i in range(1, self.HEIGHT - 1):
            self.state[i][0] = self.WALL
            self.state[i][-1] = self.WALL

        # choose a random starting position
        start_pos = None
        look_dir = random.randint(1, 4)
        match look_dir:
            case 1:  # up
                start_pos = (random.randint(2, self.HEIGHT - start_length - 2),
                             random.randint(2, self.WIDTH - 3))
                for i in range(1, start_length):
                    self.state[start_pos[0] + i, start_pos[1]] = self.BODY
                    self.body.insert(0, (start_pos[0] + i, start_pos[1]))
            case 2:  # right
                start_pos = (random.randint(2, self.HEIGHT - 3),
                             random.randint(start_length + 1, self.WIDTH - 2))
                for i in range(1, start_length):
                    self.state[start_pos[0], start_pos[1] - i] = self.BODY
                    self.body.insert(0, (start_pos[0], start_pos[1] - i))
            case 3:  # down
                start_pos = (random.randint(start_length + 1, self.HEIGHT - 2),
                             random.randint(2, self.WIDTH - 3))
                for i in range(1, start_length):
                    self.state[start_pos[0] - i, start_pos[1]] = self.BODY
                    self.body.insert(0, (start_pos[0] - i, start_pos[1]))
            case 4:  # left
                start_pos = (random.randint(2, self.HEIGHT - 3),
                             random.randint(2, self.WIDTH - start_length - 2))
                for i in range(1, start_length):
                    self.state[start_pos[0], start_pos[1] + i] = self.BODY
                    self.body.insert(0, (start_pos[0], start_pos[1] + i))

        # add head
        self.body.append(start_pos)
        self.state[start_pos[0], start_pos[1]] = self.HEAD

        self.generate_food(foods)
        if self.is_render: self.render()
        return np.copy(self.state)

    def step(self, action: int) -> tuple[NDArray[float], int, bool]:
        reward = self.rewards['survive']
        done = False
        match action:
            case 0:  # straight
                self.body.append((2 * self.body[-1][0] - self.body[-2][0], 2 * self.body[-1][1] - self.body[-2][1]))
            case 1:  # left
                if self.body[-1][1] != self.body[-2][1]:
                    self.body.append((self.body[-1][0] + self.body[-1][1] - self.body[-2][1], self.body[-1][1]))
                else:
                    self.body.append((self.body[-1][0], self.body[-1][1] + self.body[-2][0] - self.body[-1][0]))
            case 2:  # right
                if self.body[-1][0] != self.body[-2][0]:
                    self.body.append((self.body[-1][0], self.body[-1][1] + self.body[-1][0] - self.body[-2][0]))
                else:
                    self.body.append((self.body[-1][0] + self.body[-2][1] - self.body[-1][1], self.body[-1][1]))

        match self.state[self.body[-1][0], self.body[-1][1]]:
            case self.EMPTY:
                if len(self.foods) == 1:
                    diff = (np.sqrt(
                        (self.body[-2][0] - self.foods[0][0]) ** 2 + (self.body[-2][1] - self.foods[0][1]) ** 2) -
                            np.sqrt((self.body[-1][0] - self.foods[0][0]) ** 2 + (
                                    self.body[-1][1] - self.foods[0][1]) ** 2))
                    if diff > 0:
                        reward = self.rewards['move closer']
                    elif diff < 0:
                        reward = self.rewards['move further']
                    else:
                        reward = self.rewards['survive']
                else:
                    reward = self.rewards['survive']
                self.state[self.body[-1][0], self.body[-1][1]] = self.HEAD
                self.state[self.body[-2][0], self.body[-2][1]] = self.BODY
                self.state[self.body[0][0], self.body[0][1]] = self.EMPTY
                self.body.pop(0)
                self.loop_counter += 1
                if self.loop_counter == self.loop_threshold:
                    done = True
            case self.FOOD:
                self.loop_counter = 0
                reward = self.rewards['food']
                self.state[self.body[-1][0], self.body[-1][1]] = self.HEAD
                self.state[self.body[-2][0], self.body[-2][1]] = self.BODY
                self.foods.remove(self.body[-1])
                self.generate_food(1)
                self.score += 1
            case self.BODY:
                reward = self.rewards['hit body']
                done = True
            case self.WALL:
                reward = self.rewards['hit wall']
                done = True
        self.episode_length += 1
        self.total_reward += reward
        if self.is_render:
            self.render()
            self.clock.tick(self.fps)
        return np.copy(self.state), reward, done

    def generate_food(self, num_foods: int = 1) -> None:
        for i in range(num_foods):
            pos = random.randint(0, self.HEIGHT - 1), random.randint(0, self.WIDTH - 1)
            while self.state[pos[0], pos[1]] != 0:
                pos = random.randint(0, self.HEIGHT - 1), random.randint(0, self.WIDTH - 1)
            self.state[pos[0], pos[1]] = self.FOOD
            self.foods.append(pos)

    def close(self):
        if self.is_render:
            pygame.quit()


class VecSnake:
    def __init__(self, shape: tuple[int, int, int], loop_threshold: int = -1, continuous: bool = True,
                 render: bool = True):
        """

        :param shape: (height, width, envs)
        :param loop_threshold: negative value = no loop declaration
        :param continuous: whether to reset done envs without waiting for all envs
        :param render:
        """
        self.size = shape[2]
        self.envs = [Snake((shape[0], shape[1]), loop_threshold, render) for i in range(self.size)]
        self.continuous = continuous
        if not self.continuous:
            self.active_envs = self.envs.copy()

    def reset_one(self, index: int, start_length: int, foods: int) -> NDArray[float]:
        return self.envs[index].reset(start_length, foods)

    def reset_all(self, start_lengths: list[int], foods: list[int]) -> NDArray[float]:
        return np.array([self.envs[i].reset(start_lengths[i], foods[i]) for i in range(self.size)])

    def step(self, actions: list[int]) -> tuple[list[NDArray[float]], list[int], list[bool]]:
        new_states = []
        rewards = []
        dones = []
        for i in range(len(actions)):
            if self.continuous:
                feedback = self.envs[i].step(actions[i])
            else:
                feedback = self.active_envs[i].step(actions[i])
            new_states.append(feedback[0])
            rewards.append(feedback[1])
            dones.append(feedback[2])

        return new_states, rewards, dones

    def get_states(self) -> NDArray[float]:
        return np.array([np.copy(self.envs[i].state) for i in range(self.size)])

    def scores(self, index: int | None = None) -> list[int] | int:
        if index is None:
            return [env.score for env in self.envs]
        return self.envs[index].score

    def total_rewards(self, index: int | None = None) -> list[int] | int:
        if index is None:
            return [env.total_reward for env in self.envs]
        return self.envs[index].total_reward

    def episode_lengths(self, index: int | None = None) -> list[int] | int:
        if index is None:
            return [env.episode_length for env in self.envs]
        return self.envs[index].episode_length

    def close_one(self, index):
        if self.continuous:
            self.envs[index].close()
        else:
            self.active_envs[index].close()

    def close_all(self):
        for i in range(self.size):
            self.envs[i].close()


class VecSnake3:
    def __init__(self, shape: tuple[int, int, int], loop_threshold: int = -1, continuous: bool = True,
                 render: bool = True):
        """

        :param shape: (height, width, envs)
        :param loop_threshold: negative value = no loop declaration
        :param continuous: whether to reset done envs without waiting for all envs
        :param render:
        """
        self.size = shape[2]
        self.envs = [Snake3((shape[0], shape[1]), loop_threshold, render) for i in range(self.size)]
        self.continuous = continuous
        if not self.continuous:
            self.active_envs = self.envs.copy()

    def reset_one(self, index: int, start_length: int, foods: int) -> NDArray[float]:
        return self.envs[index].reset(start_length, foods)

    def reset_all(self, start_lengths: list[int], foods: list[int]) -> NDArray[float]:
        return np.array([self.envs[i].reset(start_lengths[i], foods[i]) for i in range(self.size)])

    def step(self, actions: list[int]) -> tuple[list[NDArray[float]], list[int], list[bool]]:
        new_states = []
        rewards = []
        dones = []
        for i in range(len(actions)):
            if self.continuous:
                feedback = self.envs[i].step(actions[i])
            else:
                feedback = self.active_envs[i].step(actions[i])
            new_states.append(feedback[0])
            rewards.append(feedback[1])
            dones.append(feedback[2])

        return new_states, rewards, dones

    def get_states(self) -> NDArray[float]:
        return np.array([np.copy(self.envs[i].state) for i in range(self.size)])

    def scores(self, index: int | None = None) -> list[int] | int:
        if index is None:
            return [env.score for env in self.envs]
        return self.envs[index].score

    def total_rewards(self, index: int | None = None) -> list[int] | int:
        if index is None:
            return [env.total_reward for env in self.envs]
        return self.envs[index].total_reward

    def episode_lengths(self, index: int | None = None) -> list[int] | int:
        if index is None:
            return [env.episode_length for env in self.envs]
        return self.envs[index].episode_length

    def close_one(self, index):
        if self.continuous:
            self.envs[index].close()
        else:
            self.active_envs[index].close()

    def close_all(self):
        for i in range(self.size):
            self.envs[i].close()
