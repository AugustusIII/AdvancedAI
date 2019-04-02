from math import *
import random
import sys
import pygame

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import tensorflow
import keras
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras import activations
from keras.models import load_model

# Display
dx = 920
dy = 480
rad = 20
gLev = 400

WIN = pi*dx/8+dx/2 - 10
LOSE = -pi*dx/8+dx/2 - 10

red = (255,0,0)
sky = (180,225,255)
earth = (149,69,53)
star = (255,230,20)
green = (0,120,0)
black = (0,0,0)

cvals = [((x*pi/100),cos(x*pi/100)) for x in range(-100,100,1)]
curve = [(x*dx/8 + dx/2,(y-1)*dy/4 + gLev) for (x,y) in cvals]

# IMPORTANT CONSTANTS
dt = 500
gamma = 1.5
g = 1
Episodes = 30000

num_actions = 2

class Game(object):
    def __init__(self):
        self.acc = 0.000001
        self.gc = 0.0000015 * g
        self.reset()

    def reset(self):
        self.pos = 0.0
        self.vel = 0.0
        self.gametime = 0

        S = np.array([self.pos, self.vel])
        return S

    def update(self, A, dt):
        R = -dt/1000
        end = False
        self.gametime += dt

        if A == 0:
            self.vel -= self.acc * dt
        if A == 1:
            self.vel += self.acc * dt       # Add control force

        self.vel -= self.gc*sin(self.pos) * dt  # Gravity
        self.vel -= self.vel * dt * 0.0001  # Friction
        self.pos += self.vel * dt           # Update position

        if self.pos >= pi:
            R = 10.0
            end = True
        if self.pos <= -pi:
            R = -10.0
            end = True
        if self.gametime >= 10000:
            end = True

        if end:
            self.reset()

        S = np.array([self.pos, self.vel])

        return S, R, end

class Agent(object):
    def __init__(self):
        self.model = load_model('mc-pg-g_12-dt_500.h5')

    def getAction(self, S):
        S = np.expand_dims(S, axis=0)
        probs = np.squeeze(self.model.predict(S))
        sample = np.random.choice(np.arange(num_actions), p=probs)

        return sample

# Stats keeping track of agent performance
matplotlib.style.use('ggplot')
stats_scores = np.zeros(Episodes)
stats_lengths = np.zeros(Episodes)


#initialize display
pygame.init()
screen = pygame.display.set_mode((dx,dy))
clock = pygame.time.Clock()

# Initialize the game and the agents
g = Game()
agent = Agent()

for e in range(Episodes):
    s = g.reset()
    total_score = 0
    S = []
    A = []
    G = []

    for t in range(1,1000):

        dt = clock.tick(2)

        screen.fill(sky)
        pygame.draw.rect(screen, earth, (0,gLev,dx,dy-gLev), 0)
        pygame.draw.lines(screen, black, False, curve, 3)
        pygame.draw.ellipse(screen, star, (WIN, -dy/2 + gLev - 40 ,20,40), 0)
        pygame.draw.ellipse(screen, red, (LOSE, -dy/2 + gLev - 40 ,20,40), 0)

        # QUIT GAME
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        pygame.draw.circle(screen, green, (int(g.pos*dx/8 + dx/2), \
                                     int((cos(g.pos)-1)*dy/4 + gLev - rad)), rad, 0)
        pygame.display.update()

        a = agent.getAction(s)
        S.append(s)
        A.append(a)
        s, r, end = g.update(a, dt)
        G.append(r)
        total_score += r

        #print(S)

        if end:
            acc = 0
            for i in reversed(range(len(G))):
                acc = gamma * acc + G[i]
                G[i] = acc

            A = np_utils.to_categorical(A, num_classes=num_actions)

            print("Game:", e, "completed in:", t, ", earning:", "%.2f"%total_score, "points.")
            stats_scores[e] = total_score
            stats_lengths[e] = t
            break

window = 100
score_ave = np.convolve(stats_scores, np.ones((window,))/window, mode="valid")
t_ave = np.arange(score_ave.size)
plt.rcParams['figure.figsize'] = [15, 5]
plt.plot(t_ave, score_ave)
plt.show()