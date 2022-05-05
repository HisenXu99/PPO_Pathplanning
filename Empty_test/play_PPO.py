# -*- coding: UTF-8 -*-
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import os
import sys
import math
from PPO_A2C import PPO
from Environment_Empty import envmodel
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 默认显卡0

env = envmodel()

# 动作指令集---> v,w
action_dict = {0: [1.0, -1.0], 1: [1.0, -0.5], 2: [1.0, 0.0],
               3: [1.0, 0.5], 4: [1.0, 1.0], 5: [0.5, -1.0],
               6: [0.5, 0.0], 7: [0.5, 1.0], 8: [0.0, -1.0],
               9: [0.0, 0.0], 10: [0.0, 1.0]}


#TODO：改变学习率，从batch中抽几帧，改变sleep时间，改连续变为离散

class play:
    def __init__(self):
        # Get parameters
        self.progress = ''
        self.Num_action = len(action_dict)
        
        self.date_time = str(datetime.date.today())

        self.loadpath= 'Empty/saved_networks/practice/2022-04-14-03_22_05'

        # Initial parameters
        # ------------------------------
        self.Num_start_training = 0
        self.Num_training = 600000
        # ------------------------------
        self.Num_test = 0
        self.GAMMA = 0.9
        # self.Final_epsilon = 0.1
        # ------------------------------
        # self.Epsilon = 0.5
        # ------------------------------
       
        self.step = 1
        self.score = 0
        self.episode = 0
        # ------------------------------

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        # parameters for skipping and stacking
        self.Num_skipFrame = 1
        self.Num_stackFrame = 4

        # Parameters for network
        self.img_size = 80  # input image size

        # Initialize agent robot
        self.agentrobot = 'jackal0'

        # Define the distance from start point to goal point
        self.d = 15.0

        # Define the step for updating the environment
        self.MAXSTEPS = 300
        # ------------------------------
        self.MAXEPISODES = 5000
        # ------------------------------

        # Parameter for LSTM
        self.Num_dataSize = 364  # 360 sensor size + 4 self state size

        self.Num_batch = 16

        self.LR=0.0001

        # sess, self.saver = self.init_sess()
        self.PPO=PPO(self.loadpath,self.LR)
        pass



        # Initialize input
    def input_initialization(self, env_info):
        state = env_info[0]  # laser info + self state
        state_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.Num_dataSize))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame,
                        :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        observation = env_info[1]  # image info
        observation_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            observation_set.append(observation)
        # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros(
            (self.img_size, self.img_size, self.Num_stackFrame))
        # print("shape of observation stack={}".format(observation_stack.shape))
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 -
                                                                   (self.Num_skipFrame * stack_frame)]
        observation_stack = np.uint8(observation_stack)

        return observation_stack, observation_set, state_stack, state_set


    # Resize input information
    def resize_input(self, env_info, observation_set, state_set):
        observation = env_info[1]
        observation_set.append(observation)
        # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros(
            (self.img_size, self.img_size, self.Num_stackFrame))
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 -
                                                                   (self.Num_skipFrame * stack_frame)]
        del observation_set[0]
        observation_stack = np.uint8(observation_stack)

        state = env_info[0]
        state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.Num_dataSize))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame,
                        :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        del self.state_set[0]
        return observation_stack, observation_set, state_stack, state_set



    def select_action(self, state_stack,image):
        action=self.PPO.choose_action([state_stack],[image])
        return action


    def main(self):
        reward_list = []
        # 随机种子
        random.seed(1000)
        np.random.seed(1000)
        tf.set_random_seed(1234)
        # 随机初始化起点和终点的位置
        while(True):
            randposition = 2 * self.d * \
                np.random.random_sample((2, 2)) - self.d
            if math.sqrt((randposition[0][0]-randposition[1][0])**2+(randposition[0][1]-randposition[1][1])**2) > 20.0:
                break
        env.reset_env(start=[randposition[0][0], randposition[0][1]], goal=[randposition[1][0], randposition[1][1]])
        env_info = env.get_env()
        # env.info为4维，第1维为相机消息，第2维为agent robot的self state，第3维为terminal，第4维为reward
        observation_stack, self.observation_set, state_stack, self.state_set= self.input_initialization(env_info)
        state_stack_x=env_info[0][-4:]
        step_for_newenv = 0


        # Training & Testing
        while True:
            # Select Actions 根据进度选取动作
            action = self.select_action( state_stack_x,observation_stack)
            env.step(action_dict[action])
            env_info = env.get_env()
            a = np.zeros([self.Num_action])
            a[action]=1
            next_observation_stack, self.next_observation_set, next_state_stack, self.next_state_set= self.resize_input(env_info, self.observation_set, self.state_set)  # 调整输入信息
            next_state_stack_x=env_info[0][-4:]

            terminal = env_info[-2]  # 获取terminal
            reward = env_info[-1]  # 获取reward

            if step_for_newenv == self.MAXSTEPS:
                terminal = True




            # If progress is finished -> close!
            if  self.episode == self.MAXEPISODES:
                print('Finished!!')
                break

            # Update information
            self.step += 1
            self.score += reward
            observation_stack = next_observation_stack
            state_stack_x = next_state_stack_x
            step_for_newenv = step_for_newenv + 1


            # If terminal is True
            if terminal == True:

                step_for_newenv = 0
                # Print informations
                print('step:'+str(self.step)+'/'+'episode:'+str(self.episode)+'/'+'progress:' +self.progress+'/'+'/'+'score:' + str(self.score))
                
                reward_list.append(self.score)
                reward_array = np.array(reward_list)
                self.episode += 1

                self.score = 0

                # Initialize game state
                # 随机初始化起点和终点的位置
                while(True):
                    randposition = 2 * self.d * \
                        np.random.random_sample((2, 2)) - self.d
                    if math.sqrt((randposition[0][0]-randposition[1][0])**2+(randposition[0][1]-randposition[1][1])**2) > 20.0:
                        break
                env.reset_env(start=[randposition[0][0], randposition[0][1]], goal=[
                              randposition[1][0], randposition[1][1]])
                env_info = env.get_env()
                state_stack_x=env_info[0][-4:]
        pass


if __name__ == '__main__':
    agent = play()
    agent.main()