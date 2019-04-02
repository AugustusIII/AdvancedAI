from math import *
import random
import sys

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


# IMPORTANT CONSTANTS
dt = 500
gamma = 1.0
g = 1
Episodes = 20000

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
        self.model = self.getModel()
        self.train = self.getTrain()

    def getModel(self):
        #Set up the Model
        X = layers.Input(shape=(2,))
        net = X
        net = layers.Dense(256)(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(128)(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(64)(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(num_actions)(net)
        net = layers.Activation("softmax")(net)
        
        model = Model(inputs=X, outputs=net)

        return model

    def getTrain(self):
        prob_placeholder = self.model.output
        action_placeholder = K.placeholder(shape=(None, num_actions),
                                                  name="action_onehot")
        reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        action_prob = K.sum(prob_placeholder * action_placeholder, axis=1)
        log_prob = K.log(action_prob)

        loss = - log_prob * reward_placeholder
        loss = K.mean(loss)

        rms = optimizers.RMSprop()

        updates = rms.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        train = K.function(inputs=[self.model.input,
                                           action_placeholder,
                                           reward_placeholder],
                                   outputs=[],
                                   updates=updates)

        return train

    def getAction(self, S):
        S = np.expand_dims(S, axis=0)
        probs = np.squeeze(self.model.predict(S))
        sample = np.random.choice(np.arange(num_actions), p=probs)

        return sample

    def fit(self, S, A, G):
    	acc = 0
    	for i in reversed(range(len(G))):
            acc = gamma * acc + G[i]
            G[i] = acc

    	A = np_utils.to_categorical(A, num_classes=num_actions)

    	self.train([S,A,G])

def getStats():
    # Stats keeping track of agent performance
    stats_scores = np.zeros(Episodes)

    # Initialize the game and the agents
    g = Game()
    agent = Agent()

    for e in range(Episodes):
        s = g.reset()
        total_score = 0
        S = []
        A = []
        G = []

        for t in range(1,100000):
            a = agent.getAction(s)
            S.append(s)
            A.append(a)
            s, r, end = g.update(a, dt)
            G.append(r)
            total_score += r

            if end:
                agent.fit(S,A,G)
	
                print("Game:", e, "completed in:", t, ", earning:", "%.2f"%total_score, "points.")
                stats_scores[e] = total_score
                break
    agent.model.save('mc-pg-g_1-dt_500.h5')
    return stats_scores

stats = []
it = 3
for i in range(it):
    stats.append(getStats())

ave = sum(np.array(stats))/it

final = np.convolve(ave, np.ones((100,))/100, mode="valid")
np.save('mc-pg-g_1-dt_500', final)

#np.save('mountain-sarsa-scores', stats_scores)

#agent.model.save('pg-climb.h5')
#
#window = 100
#score_ave = np.convolve(stats_scores, np.ones((window,))/window, mode="valid")
#t_ave = np.arange(score_ave.size)
#plt.rcParams['figure.figsize'] = [15, 5]
#plt.plot(t_ave, score_ave)
#plt.show()