import os
import time

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hide pygame's welcome output
import pygame
import logs
from environments import Snake

# Only modify these 2 lines:
fps = 20
height, width = 15, 15

# Load game recording
game = logs.read_log(f'{height}x{width} best game')
print('score:', game[0])
print('shape:', game[1].shape)
# Initialize variables
env = Snake(game[1].shape)
clock = pygame.time.Clock()
flag = True
# Iterate over frames
for frame in game[1:]:
    env.state = frame
    env.render()
    clock.tick(fps)
    pygame.event.get()  # Clean user input
    # Wait on first frame before starting
    if flag:
        time.sleep(2)
        flag = False
time.sleep(3)  # Wait on final frame
