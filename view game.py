import os
import time

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import logs
from environments import Snake

fps = 30
game = logs.read_log('30x30 best game')
print('score:', game[0])
print('shape:', game[1].shape)
env = Snake(game[1].shape)
clock = pygame.time.Clock()
flag = True
for frame in game[1:]:
    env.state = frame
    env.render()
    clock.tick(fps)
    pygame.event.get()
    if flag:
        time.sleep(5)
        flag = False
time.sleep(5)
