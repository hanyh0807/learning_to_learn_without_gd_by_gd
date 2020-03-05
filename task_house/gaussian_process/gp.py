# -*- coding: utf-8 -*-
"""
@author: Younghyun Han
"""
import os

import numpy as np
from multiprocessing import Process, Queue, Array, Value, Event
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class simul():
    def __init__(self, start_process = True, dummy_y_house=None, test_y=None, y_scaling=True):
        self.y_scaling = y_scaling
        self.dummy_y_house = dummy_y_house
        if dummy_y_house is not None:
            self.max_task_num = self.dummy_y_house.shape[0]

        self.interval=0.1
        self.test_y = test_y
        self.cur_test_y = []
        
        self.rq = Queue()
        self.params = Array('d', int(np.sum(self.get_num_params())))
        self.tvt = Value('d', 0) # 0 - train, 1 - val, 2 - test
        self.seed = Value('d', 0)
        self.do_reset_initials = Value('d',0)
        self.do_initialize_task = Value('d',0)
        self.START_CALC_EVENT = Event()
        self.START_CALC_EVENT.clear()

        if start_process:
            self.ptrain = Process(target=self.return_reward_notmut)
            self.ptrain.start()
            
    def stop_process(self):
            self.ptrain.terminate()

    def get_num_params(self):
        param = 2
        node = 0
        return param, node

    def get_link_scale(self, scale=5):
        
        numpoints = int(scale/self.interval)
        action_width = numpoints
        action_center = 0
        
        return action_width, action_center
    
    def initialize_task_gp(self):
        np.random.seed()

        newseed = np.random.choice(self.max_task_num)
        self.dummy_y = self.dummy_y_house[newseed].copy()

        if len(self.test_y.shape) < 3:
            self.cur_test_y = self.test_y.copy()
        else:
            newseed = np.random.choice(self.test_y.shape[0])
            self.cur_test_y = self.test_y[newseed].copy()

    def return_reward_notmut(self):
        while True:
            self.START_CALC_EVENT.wait()
            self.START_CALC_EVENT.clear()
            if self.do_initialize_task.value == 1:
                self.initialize_task_gp()
            a_pred = np.array(self.params[:].copy()).astype(np.int)
            a_pred = np.clip(a_pred, 0, self.dummy_y.shape[0]-1)

            if self.tvt.value == 0:
                data_mini = self.cur_test_y
            elif self.tvt.value == 1:
                data_mini = self.cur_test_y
            elif self.tvt.value == 2:
                data_mini = self.cur_test_y
            elif self.tvt.value == 3:   # for dummy task
                data_mini = self.dummy_y
            else:
                print('ERROR!!!!')
                exit(1)
            
            self.rq.put(np.array(data_mini[a_pred[0], a_pred[1]]))
        
