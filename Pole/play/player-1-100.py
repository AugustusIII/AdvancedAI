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
gLev = 370

red = (255,0,0)
sky = (180,225,255)
earth = (149,69,53)
star = (255,230,20)
grass = (0,127,50)
black = (0,0,0)
grey = (127,127,127)

# THE IMPORTANT CONSTANTS
dt = 100
g = 1
gamma = 1.0
Episodes = 20000

num_actions = 2

class Game(object):
    def __init__(self):
        self.theta = 0.0
        self.g = 0.000001 * g
        self.F = 0.000003
        self.l = 150
        self.reset()

    def reset(self):
        self.x = 0.0
        self.xd = 0.0
        self.h = 0.1
        self.hd = 0.0
        self.gametime = 0

        S = np.array([self.x, self.xd, self.h, self.hd])

        return S

    def update(self, A, dt):
        R = dt/1000
        end = False
        self.gametime += dt

        xdd = 0
        if A == 0 and self.x > -0.5:	# Control acceleration
            xdd = -self.F
        if A == 1 and self.x < 0.5:
            xdd = self.F

        self.theta = asin(self.h)
        self.hd += (self.g*sin(self.theta)*cos(self.theta) - xdd*cos(self.theta)*cos(self.theta)) * dt

        # Integrate position and speeds over timestep
        self.xd += xdd * dt
        self.x += self.xd * dt
        self.h += self.hd * dt

        # Clip values to prevent undefined behavior
        if self.h > 1:
            self.h = 1
        if self.h < -1:
            self.h = -1

        if self.x >= 0.5:
            self.x = 0.5
            self.xd = 0.0
        if self.x <= -0.5:
            self.x = -0.5
            self.xd = 0.0

        # End conditions
        if self.h <= -1.0 or self.h >= 1.0:
            R = -100
            end = True

        if self.gametime >= 50000:
            end = True

        if end:
            self.reset()

        S = np.array([self.x, self.xd, self.h, self.hd])

        return S, R, end

class Agent(object):
    def __init__(self):
        self.model = load_model('mc-pg-g_1-dt_100.h5')

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
        dt = clock.tick(10)

        screen.fill(sky)
        pygame.draw.rect(screen, grass, (0,gLev+30,dx,dy-gLev+30), 0)
        pygame.draw.rect(screen, earth, (dx/4-30,gLev,dx/2+60,30), 0)
        pygame.draw.rect(screen, earth, (dx/4-60,gLev-30,30,60), 0)
        pygame.draw.rect(screen, earth, (3*dx/4+60 - 30,gLev-30,30,60), 0)

        xcor = int((g.x+1)*dx/2)
        ball = (int(xcor + g.l*cos(pi/2 - g.theta)), int((gLev-30) - g.l*sin(pi/2 - g.theta)))

        pygame.draw.rect(screen, grey, (xcor-30,gLev-30,60,30), 0)
        pygame.draw.circle(screen, red, ball, 20, 0)
        pygame.draw.line(screen, black, (xcor,gLev-15), ball, 3)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

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