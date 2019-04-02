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
dt = 500
g = 1
gamma = 0.5
Episodes = 10000

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
        model.add(Dense(256, input_dim=2, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(128, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(64, kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(num_actions, activation='linear'))
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

def getStats():
    # Stats keeping track of agent performance
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

    agent.model.save('mc-sarsa-g_05-dt_5000.h5')
    return stats_scores

stats = []
it = 3
for i in range(it):
    stats.append(getStats())

ave = sum(np.array(stats))/it

final = np.convolve(ave, np.ones((100,))/100, mode="valid")
np.save('mc-pg-g_05-dt_500', final)

#np.save('mountain-sarsa-scores', stats_scores)
#
#window = 100
#score_ave = np.convolve(stats_scores, np.ones((window,))/window, mode="valid")
#t_ave = np.arange(score_ave.size)
#plt.rcParams['figure.figsize'] = [15, 5]
#plt.plot(t_ave, score_ave)
#plt.show()