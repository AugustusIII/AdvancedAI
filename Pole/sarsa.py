# Import librarie
from math import *
import random

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import activations

# THE IMPORTANT CONSTANTS
dt = 100
g = 1
gamma = 0.9
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
        if A == 0 and self.x > -0.5:    # Control acceleration
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

        #if self.gametime >= 50000:
        #    end = True

        if end:
            self.reset()

        S = np.array([self.x, self.xd, self.h, self.hd])

        return S, R, end

class Agent(object):
    def __init__(self):
        self.model = self.getModel()
        self.target_model = self.getModel()
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decr = 0.0
        self.tao = 1
        self.tao_max = 100
        self.tao_incr = 0.1
        self.counter = 0

    def getModel(self):
        model = Sequential()
        model.add(Dense(32, input_dim=4, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(32, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(16, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['mae', 'acc'])
        
        return model
    
    def train(self,s,a,s_p,a_p,r,end):
        s = np.expand_dims(s, axis=0)
        s_p = np.expand_dims(s_p, axis=0)
        target = self.model.predict(s)[0]
        
        if end:
            target[a] = r
        else:
            target[a] = r + gamma*self.target_model.predict(s_p)[0][a_p]
        
        target = np.expand_dims(target, axis=0)
        self.model.fit(s, target, epochs=1, verbose=0)

        self.counter += 1
        if self.counter >= 100:
            self.counter = 0
            self.target_model.set_weights(self.model.get_weights()) 
        

    def getAction_eps(self, S):
        S = np.expand_dims(S, axis=0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(num_actions)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decr
        
        q_vals = self.model.predict(S)[0]
        return np.argmax(q_vals)

    def getAction_softmax(self, S):
        S = np.expand_dims(S, axis=0)
        q_vals = np.squeeze(self.model.predict(S)) * self.tao
        probs = np.squeeze(np.exp(q_vals) / np.sum(np.exp(q_vals), axis=0))
        try:
            sample = np.random.choice(np.arange(num_actions), p=probs)
        except:
            print("Both 0")
            sample = 0

        if self.tao > self.tao_max:
            self.tao += self.tao_incr
        return sample

# Stats keeping track of agent performance
matplotlib.style.use('ggplot')
stats_scores = np.zeros(Episodes)

# Initialize the game and the agents
g = Game()
agent = Agent()

for e in range(Episodes):
    s = g.reset()
    a = agent.getAction_eps(s)
    a_p = None
    total_score = 0

    for t in range(1,1000):

        s_p, r, end = g.update(a, dt)
        total_score += r

        if not end:
            a_p = agent.getAction_eps(s_p)
            agent.train(s,a,s_p,a_p,r,end)
            
            s = s_p
            a = a_p

        if end:
            agent.train(s,a,s_p,a_p,r,end)
            print("Game:", e, "completed in:", t, ", earning:", "%.2f"%total_score, "points.")
            stats_scores[e] = total_score
            break

#np.save('pole-sarsa-scores', stats_scores)
#
#window = 100
#score_ave = np.convolve(stats_scores, np.ones((window,))/window, mode="valid")
#t_ave = np.arange(score_ave.size)
#plt.rcParams['figure.figsize'] = [15, 5]
#plt.plot(t_ave, score_ave)
#plt.show()


