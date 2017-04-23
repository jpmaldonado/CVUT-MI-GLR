#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 14:13:18 2017

@author: pablo
"""


import json
import gym 
import skimage
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers.core import Dense,Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam


class AtariEnvironment(object):
    
    def __init__(self, gym_env, resized_width, resized_height, img_frames):
        self.env = gym_env
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.img_frames = img_frames
        self.gym_actions = range(gym_env.action_space.n)    
        
        if (gym_env.spec.id == "Pong-v0" or gym_env.spec.id == "Breakout-v0"):
            # Gym returns 6 possible actions for breakout and pong.
            # Only three are used, the rest are no-ops. This just lets us
            # pick from a simplified "LEFT", "RIGHT", "NOOP" action space.
            self.gym_actions = [1,2,3]

    def get_initial_state(self):
        self.state_buffer = deque()
        x = self.env.reset()        
        x = self.preprocess_frame(x)
        s = np.stack((x,x,x,x), axis = 0)
        
        for i in range(self.img_frames-1):
            self.state_buffer.append(x)
            
        # Reshape for Keras            
        s = s.reshape(1, s.shape[0], s.shape[1], s.shape[2]) #1*80*80*4            
        return s

    def preprocess_frame(self,image):
        image = skimage.color.rgb2gray(image)
        image = skimage.transform.resize(image,(80,80), mode ='constant')
        return image
    
    def step(self, action):
        x1, r, d, _ = self.env.step(self.gym_actions[action])
        x1 = self.preprocess_frame(x1)
        previous_frames = np.array(self.state_buffer)
        s1 = np.empty((self.img_frames, self.resized_height, self.resized_width))
        s1[:self.img_frames-1, ...] = previous_frames
        s1[self.img_frames-1] = x1
        
        # Reshape for Keras            
        s1 = s1.reshape(1, s1.shape[0], s1.shape[1], s1.shape[2]) #1*80*80*4
        
        # Pop the oldest frame
        self.state_buffer.popleft()
        self.state_buffer.append(x1)
        return s1, r, d

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4,4),border_mode='same',input_shape=(4,80,80)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2,2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(3))
   
    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def train(gym_env, model, n_observe, n_train, batch_size):
    atari = AtariEnvironment(gym_env,80,80,4)
    
    D = deque()

    s = atari.get_initial_state()
    j = 0
    # Get some experience playing
    #while True:
    while j<10000:        
        loss = 0
        r = 0
        Q_sa = 0

        # Play randomly (GLIE)
        if j % atari.img_frames == 0:
            if random.random()<1/(j+1):
                a = int(random.uniform(1,3))
            else: 
                q = model.predict(s)
                a = np.argmax(q)
                
        s1, r, d = atari.step(a)
        D.append((s,a,r,s1,d))
        
        if d:
            s1 = atari.get_initial_state()

        # Keep only a part of the previous moves        
        if(len(D)>5000):
            D.popleft()
#    
        if j>n_observe:
            minibatch = random.sample(D,batch_size) 
            inputs = np.zeros((batch_size,s.shape[1], s.shape[2], s.shape[3]))
            targets = np.zeros((batch_size,3))
            
            
            for i in range(0,len(minibatch)):
                s_t = minibatch[i][0]
                a_t = minibatch[i][1]
                r_t = minibatch[i][2]
                s_t1 = minibatch[i][3]
                is_terminal = minibatch[i][4]
                
                inputs[i:i+1] = s_t
                targets[i] = model.predict(s_t)
                
                Q_sa = model.predict(s_t1)
                if is_terminal:
                    targets[i,a_t] = r_t                    
                else:
                    targets[i,a_t] = r_t+0.7*np.max(Q_sa)
             
                loss += model.train_on_batch(inputs,targets)
        s = s1
        j+=1
        
        # save progress every 10000 iterations
        if j % 1000 == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        mode = ""
        if j <= n_observe:
            mode = "observe"
        elif j > n_observe and j <= n_observe + n_train:
            mode = "explore"
        else:
            mode = "train"

        print("Timestep", j, "/ Mode", mode, "/ ACTION", a, "/ REWARD", r, "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
                
if __name__ == "__main__":
    problem = 'Pong-v0'
    algo_name = 'deepqn'
    env = gym.make(problem)
    env = gym.wrappers.Monitor(env, algo_name, force=True)

    model = buildmodel()
    train(env,model,32,1000,32)
    
    
    env.close()
    #gym.upload("deepqn", api_key=API_KEY, ignore_open_monitors=True)