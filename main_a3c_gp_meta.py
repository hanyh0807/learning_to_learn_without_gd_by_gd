# -*- coding: utf-8 -*-
"""
@author: Younghyun Han
"""
import os
import sys

import threading
from multiprocessing import Lock, Array, Value, Barrier
import tensorflow as tf
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from time import sleep, time
import pickle

import task_house.gaussian_process.gp as test_prob


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def OI(r):
    oi_r = np.zeros_like(r)
    
#    oi_r[0] = r[0]
    oi_r[0] = 0
    for i in range(1, r.size):
        oi_r[i] = np.max([r[i] - np.max(r[:i]), 0])
    return oi_r

def noisy_vars(from_scope, to_scope, noise_std):
    from_scopevars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_scopevars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    to_perturb_scopevars = [var for var in to_scopevars if 'gamma' not in var.name and 'beta' not in var.name and 'curiosity' not in var.name]

    op_holder = []
    for var, perturbed_var in zip(from_scopevars, to_scopevars):
        if perturbed_var in to_perturb_scopevars:
            op = tf.assign(perturbed_var, var + tf.random.normal(shape=tf.shape(var),mean=0.,stddev=noise_std))
        else:
            op = tf.assign(perturbed_var, var)
        op_holder.append(op)

    return op_holder

def reset_vars(scope, old_var):
    scopevars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    var_shapes = [i.shape for i in scopevars]

    op_holder = []
    for i,j in zip(scopevars,old_var):
        op_holder.append(i.assign(j))
    return op_holder

class AC_Network():
    def __init__(self, parameter_dict,action_width,scope,trainer,global_step):
        self.action_width = action_width
        self.a_size = parameter_dict['a_size']
        self.h_size = parameter_dict['h_size']
        self.use_noisynet = parameter_dict['use_noisynet']
        self.use_update_noise = parameter_dict['use_update_noise']
        with tf.variable_scope(scope):
            self.entph = tf.placeholder(shape=[self.a_size],dtype=tf.float32)
            self.entropy_histogram = tf.summary.histogram('entropy_histogram',self.entph)
            
            self.prevAction =  tf.placeholder(shape=[1,None,self.a_size],dtype=tf.float32) # [batch_size, trainLength, action_size(# of PKN params)]
            l = self.prevAction
            self.prevReward = tf.placeholder(shape=[None,None,1],dtype=tf.float32)
            
            lstm_input = tf.concat([l,self.prevReward],axis=2)
            #l = tf.layers.dense(l, 256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'))
            if not self.use_update_noise:
                lstm_cell1 = tf.contrib.rnn.LSTMBlockCell(self.h_size)
                lstm_cell2 = tf.contrib.rnn.LSTMBlockCell(self.h_size)
            else:
                lstm_cell1 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
                lstm_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.h_size)
            rnn_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell1, lstm_cell2])
            c_init = np.zeros((1, lstm_cell1.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell1.state_size.h), np.float32)
            self.state_init = [[c_init, h_init]]*2
            c_in1 = tf.placeholder(tf.float32, [1, lstm_cell1.state_size.c])
            h_in1 = tf.placeholder(tf.float32, [1, lstm_cell1.state_size.h])
            c_in2 = tf.placeholder(tf.float32, [1, lstm_cell2.state_size.c])
            h_in2 = tf.placeholder(tf.float32, [1, lstm_cell2.state_size.h])
            self.state_in = ((c_in1, h_in1),(c_in2, h_in2))
            state_in1 = tf.nn.rnn_cell.LSTMStateTuple(c_in1, h_in1)
            state_in2 = tf.nn.rnn_cell.LSTMStateTuple(c_in2, h_in2)
            self.rnn,self.rnn_state = tf.nn.dynamic_rnn(\
                    inputs=lstm_input,cell=rnn_cells,dtype=tf.float32,initial_state=(state_in1, state_in2),scope='meta_rnn')
            
            self.rnn = tf.reshape(self.rnn,shape=[-1,self.h_size])
            #self.rnn = self.noisy_dense(self.rnn, self.h_size*2, 'hidden_fc', bias=True, activation_fn=tf.nn.relu)
            #self.rnn = tf.layers.dropout(self.rnn, training=True)
            
            if not self.use_noisynet:
                alpha = tf.layers.dense(self.rnn, self.a_size, activation=tf.nn.softplus, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'), kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-3))
                self.alpha = alpha + 1 + 1e-7
                beta = tf.layers.dense(self.rnn, self.a_size, activation=tf.nn.softplus, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG'), kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-3))
                self.beta = beta + 1 + 1e-7
            else:
                alpha = self.noisy_dense(self.rnn, self.a_size, 'alpha', bias=False, activation_fn=tf.nn.softplus)
                self.alpha = alpha + 1 + 1e-7
                beta = self.noisy_dense(self.rnn, self.a_size, 'beta', bias=False, activation_fn=tf.nn.softplus)
                self.beta = beta + 1 + 1e-7

            self.sample_dist = tf.distributions.Beta(self.alpha, self.beta) # small alpha -> left skewed distribution
            output = self.sample_dist.sample([1])
            output = tf.clip_by_value(output, 1e-7, 1-(1e-7))
            self.policy = tf.identity(output, name="action")
            mode_output = self.sample_dist.mode()
            mode_output = tf.clip_by_value(mode_output, 1e-7, 1-(1e-7))
            self.mode_policy = tf.identity(mode_output, name='mode_action')
            
            all_log_probs = self.sample_dist.log_prob(self.policy)
            self.all_log_probs = tf.identity(all_log_probs, name="action_probs")
            
            value = tf.layers.dense(self.rnn, 1, activation=None, use_bias=False)
            self.value = tf.identity(value, name="value_estimate")

            self.action_holder = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32, name="action_holder")
 
            if scope != 'global':           
    
                self.entropy_each = self.sample_dist.entropy()
                self.entropy = tf.reduce_mean(self.entropy_each)
        
                self.log_probs = tf.reduce_sum((self.sample_dist.log_prob(self.action_holder)), axis=1, keepdims=True)
            
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
        
                self.value_loss = tf.reduce_mean(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.policy_loss = -tf.reduce_mean(self.log_probs*self.advantages[:,tf.newaxis])
                self.loss = 0.4 * self.value_loss + self.policy_loss - self.entropy * 0.001 + tf.losses.get_regularization_loss()
                #self.loss = 0.05 * self.value_loss + self.policy_loss + tf.losses.get_regularization_loss()
                        
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients, self.grad_norms = tf.clip_by_global_norm(tf.gradients(self.loss,local_vars), 5.0)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(self.gradients,global_vars),global_step=global_step)
#                self.decay_weight = [v.assign(0.999*v) for v in global_vars]


    def noisy_dense(self, x, size, name, bias=True, activation_fn=tf.identity):
        # From https://github.com/wenh123/NoisyNet-DQN/blob/master/tf_util.py
        # the function used in eq.10,11
        def f(x):
            return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        # Initializer of \mu and \sigma 
        # Sample noise from gaussian. Factorised gaussian 
#        mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),     
#                                                    maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
#        sigma_init = tf.constant_initializer(0.4/np.power(x.get_shape().as_list()[1], 0.5))
#        p = tf.random.normal(shape=[x.get_shape().as_list()[1], 1])
#        q = tf.random.normal(shape=[1, size])
#        f_p = f(p); f_q = f(q)
#        w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)
        # Independent gaussian
        mu_init = tf.random_uniform_initializer(minval=-1*np.power(3/x.get_shape().as_list()[1], 0.5),     
                                                    maxval=1*np.power(3/x.get_shape().as_list()[1], 0.5))
        sigma_init = tf.constant_initializer(0.017)
        #sigma_init = tf.constant_initializer(0.0002)
        w_epsilon = tf.random.normal(shape=[x.get_shape().as_list()[1], size])
        b_epsilon = tf.random.normal(shape=[size])
    
        # w = w_mu + w_sigma*w_epsilon
        w_mu = tf.get_variable(name + "/w_mu", [x.get_shape()[1], size], initializer=mu_init, regularizer=tf.contrib.layers.l2_regularizer(5e-3))
        #w_mu = tf.get_variable(name + "/w_mu", [x.get_shape()[1], size], initializer=mu_init)
        w_sigma = tf.get_variable(name + "/w_sigma", [x.get_shape()[1], size], initializer=sigma_init)
        w = w_mu + tf.multiply(w_sigma, w_epsilon)
        ret = tf.matmul(x, w)
        if bias:
            # b = b_mu + b_sigma*b_epsilon
            b_mu = tf.get_variable(name + "/b_mu", [size], initializer=mu_init)
            b_sigma = tf.get_variable(name + "/b_sigma", [size], initializer=sigma_init)
            b = b_mu + tf.multiply(b_sigma, b_epsilon)
            return activation_fn(ret + b)
        else:
            return activation_fn(ret)

      
class Worker():
    def __init__(self,name,parameter_dict,model_path,global_episodes,global_step,lock,shared_a,shared_r,barrier):
        self.name = "worker_" + str(name)
        self.number = name
        self.env = test_prob.simul(dummy_y_house=parameter_dict['dummy_y_house'], test_y=parameter_dict['test_y'])
        self.model_path = model_path
        self.a_size = parameter_dict['a_size']
        self.h_size = parameter_dict['h_size']
        self.max_ep = parameter_dict['max_ep']     
        self.use_update_noise = parameter_dict['use_update_noise']
        starter_learning_rate = parameter_dict['start_lr']
        self.learning_rate = learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               64*1000, 0.996, staircase=False)
#        learning_rate = tf.train.cosine_decay_restarts(starter_learning_rate, global_step,
#                                               200, t_mul=1, alpha=0.1)
        self.trainer = tf.train.AdamOptimizer(learning_rate=starter_learning_rate)
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.total_reward = []
        self.total_intrinsic = []
        self.initial_reward = -np.inf
        self.summary_writer = tf.summary.FileWriter(model_path+"/train_"+str(self.number))
        
        self.lock = lock
        self.shared_a = shared_a
        self.shared_r = shared_r
        self.barrier = barrier

        scale = 5
        link_size, node_size = self.env.get_num_params()
        self.action_width, self.action_center = self.env.get_link_scale(scale=scale)


        self.rq = self.env.rq
        self.params = self.env.params
        self.tvt = self.env.tvt
        self.seed = self.env.seed
        self.do_reset_initials = self.env.do_reset_initials
        self.do_initialize_task = self.env.do_initialize_task
        self.START_CALC_EVENT = self.env.START_CALC_EVENT
        
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(parameter_dict,self.action_width,self.name,self.trainer,global_step)
        self.update_local_ops = update_target_graph('global',self.name)        

        if self.use_update_noise:
            self.name_perturb = self.name+'_perturb'
            self.local_AC_perturb = AC_Network(parameter_dict,self.action_width,self.name_perturb,self.trainer,global_step)

            self.name_adaptive = self.name+'_adaptive'
            self.local_AC_adaptive = AC_Network(parameter_dict,self.action_width,self.name_adaptive,self.trainer,global_step)

            self.noise_start_scale = parameter_dict['noise_start_scale']
            self.distance_threshold = parameter_dict['distance_threshold']
            self.param_noise_scale = tf.get_variable("param_noise_scale"+"_"+self.name, (), initializer=tf.constant_initializer(self.noise_start_scale), trainable=False)
            self.add_noise_ops = noisy_vars(self.name, self.name_perturb, noise_std=self.param_noise_scale)
            self.add_noise_adaptive_ops = noisy_vars(self.name, self.name_adaptive, noise_std=self.param_noise_scale)
            
            self.policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.local_AC.alpha - self.local_AC_adaptive.alpha))) + \
                                    tf.sqrt(tf.reduce_mean(tf.square(self.local_AC.beta - self.local_AC_adaptive.beta)))

        
    def run_target_task(self):
        s = np.random.rand(self.a_size)

        self.do_reset_initials.value = 1
        self.do_initialize_task.value = 1

        s = np.round((s * self.action_width) - self.action_center)
        self.params[:] = s.reshape([-1])
        self.seed.value = int(self.name[-1])*5000
        self.tvt.value = 0
        self.START_CALC_EVENT.set()
        r = self.rq.get()
        self.do_reset_initials.value = 0
        self.do_initialize_task.value = 0
                    
        if np.isnan(r):
            r = np.array(-100.)
        r = r.reshape([1,1,1])
                
        state = self.local_AC.state_init #Reset the recurrent layer's hidden state
        local_AC_running = self.local_AC

        er = -np.inf
        his = []
        #for j in range(30):
        for j in range(self.max_ep):
            feed_dict = {local_AC_running.prevReward:r.reshape([1,1,1]),
                    local_AC_running.state_in[0][0]:state[0][0], local_AC_running.state_in[0][1]:state[0][1],
                    local_AC_running.state_in[1][0]:state[1][0], local_AC_running.state_in[1][1]:state[1][1]}
            ob = s
            feed_dict[local_AC_running.prevAction] = ob.reshape([1,1,-1])

            a_dist, v, state, entropy_each = sess.run([local_AC_running.mode_policy, local_AC_running.value,
                                             local_AC_running.rnn_state, local_AC_running.entropy_each],
                                      feed_dict=feed_dict)

            a_dist_scaled = (a_dist * self.action_width) - self.action_center

            self.params[:] = a_dist_scaled.reshape([-1])
            self.seed.value = int(self.name[-1])*5000
            self.tvt.value = 0
            self.START_CALC_EVENT.set()
            r_ = self.rq.get()
            #r_ -= 0.01 * np.mean(np.round(np.abs(a_dist_scaled)))

            if np.isnan(r_):
                r_ = np.array(-100.)

            s = np.round(a_dist_scaled.reshape([-1]))
            r = r_
            his.append(r)

            if er < r:
                running_action = a_dist_scaled
                er = r

        return his, running_action

    def train(self,ep_history,sess,gamma,bootstrap_value,episode_count):
        ep_history = np.array(ep_history)
                
        rewards = ep_history[:,3].copy()
        rewards_plus = rewards.tolist()
        rewards_plus.append(bootstrap_value)
        discounted_rewards = discount(np.asarray(rewards_plus),gamma)[:-1]
        discounted_rewards = discounted_rewards.reshape([-1])
        #discounted_rewards = rewards.reshape([-1])
        #if self.use_curiosity:
        #    discounted_rewards +=  ep_history[:,-1].reshape([-1])
        #advantages = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        advantages = discounted_rewards - ep_history[:,4].reshape([-1])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        '''
        # oi
        rewards = ep_history[:,3].copy()
        rewards_plus = rewards.tolist()
        rewards_plus.append(bootstrap_value)
        rewards_plus.insert(0,ep_history[0,1])
        rewoi = OI(np.asarray(rewards_plus))[1:-1].reshape([-1])
        rewards = np.asarray(rewoi)
        discounted_rewards = rewards.reshape([-1])
        advantages = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        '''

        '''
        # oi-discount
        rewards = ep_history[:,3].copy()
        rewoi = OI(rewards).reshape([-1])
        rewoi = rewoi.tolist()
        rewoi.append(bootstrap_value)
        rewoi = discount(np.asarray(rewoi), gamma)[:-1]
        rewards = np.asarray(rewoi)
        discounted_rewards = rewards.reshape([-1])
        advantages = discounted_rewards - bootstrap_value.reshape([-1])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        advantages = advantages.reshape([-1])
        '''

        state_train = self.local_AC.state_init
        
        feed_dict={self.local_AC.target_v:discounted_rewards, self.local_AC.advantages:advantages,
                   self.local_AC.prevReward:ep_history[:,1].reshape([1,self.max_ep,1]),
                   self.local_AC.action_holder:np.vstack(ep_history[:,2]).reshape([-1,self.a_size]),
                   self.local_AC.state_in[0][0]:state_train[0][0], self.local_AC.state_in[0][1]:state_train[0][1],
                   self.local_AC.state_in[1][0]:state_train[1][0], self.local_AC.state_in[1][1]:state_train[1][1]}

        feed_dict[self.local_AC.prevAction] = np.expand_dims(np.vstack(ep_history[:,0]), 0)

        ops = [self.local_AC.value_loss, self.local_AC.policy_loss, self.local_AC.entropy, self.local_AC.rnn_state]
        ops += [self.local_AC.apply_grads]

        self.lock.acquire()
        outputs = sess.run(ops, feed_dict=feed_dict)
        self.lock.release()

        v_l = outputs[0]
        p_l = outputs[1]
        e_l = outputs[2]
        out = {'vl':outputs[0], 'pl':outputs[1], 'ent':outputs[2]}
        
        return out
        
    def work(self,gamma,sess,coord,saver,total_episodes, s_init=None):
        episode_count = sess.run(self.global_episodes)
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                starttime = time()
                sess.run(self.update_local_ops)
                ep_history = []
                running_reward = []
                running_action = 0
                running_entropy = 0
                running_intrinsic = 0
                
                if s_init is None:
#                    s = np.zeros([1,1,self.a_size]) + 0.1
                    s = np.random.rand(a_size)
                else:
                    s = s_init
                
                self.do_reset_initials.value = 0
                self.do_initialize_task.value = 1

#                try:
                s = np.round((s * self.action_width) - self.action_center)
                self.params[:] = s.reshape([-1])
                self.seed.value = episode_count+int(self.name[-1])*5000
                self.tvt.value = 3
                self.START_CALC_EVENT.set()
                r = self.rq.get()
                #r -= 0.01 * np.mean(np.round(np.abs(s)))
                self.do_reset_initials.value = 0
                self.do_initialize_task.value = 0
                    
                if np.isnan(r):
                    r = np.array(-100.)
                r = r.reshape([1,1,1])
                
                state = self.local_AC.state_init #Reset the recurrent layer's hidden state

                if self.use_update_noise:                
                    if episode_count % 2 == 0:
                        #with tf.control_dependencies(self.add_noise_adaptive_ops):
                        sess.run(self.add_noise_adaptive_ops)
                        fd = {self.local_AC.prevAction:s.reshape([1,1,self.a_size]),
                                self.local_AC.prevReward:r.reshape([1,1,1]),
                                self.local_AC.state_in[0][0]:state[0][0], self.local_AC.state_in[0][1]:state[0][1],
                                self.local_AC.state_in[1][0]:state[1][0], self.local_AC.state_in[1][1]:state[1][1],
                                self.local_AC_adaptive.prevAction:s.reshape([1,1,self.a_size]),
                                self.local_AC_adaptive.prevReward:r.reshape([1,1,1]),
                                self.local_AC_adaptive.state_in[0][0]:state[0][0], self.local_AC_adaptive.state_in[0][1]:state[0][1],
                                self.local_AC_adaptive.state_in[1][0]:state[1][0], self.local_AC_adaptive.state_in[1][1]:state[1][1]
                            }
                        policy_dist = sess.run(self.policy_distance, feed_dict=fd)
                        if policy_dist < self.distance_threshold:
                            sess.run(self.param_noise_scale.assign(self.param_noise_scale * 1.01))
                        else:
                            sess.run(self.param_noise_scale.assign(self.param_noise_scale / 1.01))

                    sess.run(self.add_noise_ops)
                    local_AC_running = self.local_AC_perturb
                else:
                    local_AC_running = self.local_AC
                
                er = -np.inf
                actiontemp = []
                for j in range(self.max_ep):
                    feed_dict = {local_AC_running.prevReward:r.reshape([1,1,1]),
                            local_AC_running.state_in[0][0]:state[0][0], local_AC_running.state_in[0][1]:state[0][1],
                            local_AC_running.state_in[1][0]:state[1][0], local_AC_running.state_in[1][1]:state[1][1]}
                    ob = s
                    feed_dict[local_AC_running.prevAction] = ob.reshape([1,1,-1])

                    a_dist, v, state, entropy_each = sess.run([local_AC_running.policy, local_AC_running.value,
                                                               local_AC_running.rnn_state, local_AC_running.entropy_each],
                                      feed_dict=feed_dict)
 
                    normalized_entropy = entropy_each.reshape([-1])
                    a_dist_scaled = (a_dist * self.action_width) - self.action_center

                    self.params[:] = a_dist_scaled.reshape([-1])
                    self.seed.value = episode_count+int(self.name[-1])*5000
                    self.tvt.value = 3
                    self.START_CALC_EVENT.set()
                    r_ = self.rq.get()
                    #r_ -= 0.01 * np.mean(np.round(np.abs(a_dist_scaled)))

                    if np.isnan(r_):
                        r_ = np.array(-100.)


                    his = [ob.reshape([-1,self.a_size]),r.reshape([-1]),a_dist,r_.reshape([-1]),v[0,0]]
                    ep_history.append(his) # (s,r) - state, action - action, r_ - reward, v - value, a_dist - action_probs
                    s = np.round(a_dist_scaled.reshape([-1]))
                    r = r_
                    
                    running_reward.append(r)
                    if er < r:
                        running_action = a_dist_scaled
                        running_entropy = normalized_entropy
                        actiontemp = a_dist
                        er = r
#                print('Episode time ', time()-starttime)
                
                feed_dict = {local_AC_running.prevReward:r.reshape([1,1,1]),
                        local_AC_running.state_in[0][0]:state[0][0], local_AC_running.state_in[0][1]:state[0][1],
                        local_AC_running.state_in[1][0]:state[1][0], local_AC_running.state_in[1][1]:state[1][1]}
                ob = s
                feed_dict[local_AC_running.prevAction] = ob.reshape([1,1,-1])

                v1 = sess.run([local_AC_running.value],
                              feed_dict=feed_dict)[0]

                
                self.total_reward.append(np.sum(OI(np.array(running_reward))))
                
                if self.name == 'worker_0':
                    sess.run(self.increment)
                    if episode_count % 100 == 0:
                        maxr_arg = np.argmax(self.total_reward[-100:])
                        print(np.nanmean(self.total_reward[-100:]),
                              self.total_reward[-100:][maxr_arg], episode_count)
                    if episode_count > 16:
                        cr = np.mean(self.total_reward[-16:])
                        if self.initial_reward < cr:
                            self.initial_reward = cr
#                            saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                    #if episode_count > total_episodes:
                    #    self.env.stop_process()
                    #    coord.request_stop()

                losses = self.train(ep_history,sess,gamma,v1,episode_count)
                if episode_count % 10 == 0:
                    if episode_count % 20 == 0:
                        target_rew, target_action = self.run_target_task()
                        self.params[:] = target_action.reshape([-1])
                        self.tvt.value = 2
                        self.START_CALC_EVENT.set()
                        rtest = self.rq.get()

                        rtraining_whole = np.max(target_rew)

                    
                    mean_reward = np.mean(self.total_reward[-16:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    if episode_count % 20 == 0:
                        summary.value.add(tag='Perf/R_test', simple_value=float(rtest))
                        summary.value.add(tag='Perf/R_train_whole', simple_value=float(rtraining_whole))
                        summary.value.add(tag='Perf/R_OI_test', simple_value=float(np.sum(OI(np.array(target_rew)))))
                    if self.use_update_noise:
                        summary.value.add(tag='Noise/Mean_distance', simple_value=float(policy_dist))
                        ps = sess.run(self.param_noise_scale)
                        summary.value.add(tag='Noise/Perturb_scale', simple_value=float(ps))
                    self.summary_writer.add_summary(summary, episode_count)
                    summ = sess.run([self.local_AC.entropy_histogram], feed_dict={self.local_AC.entph:normalized_entropy.reshape([-1])})[0]
                    self.summary_writer.add_summary(summ, episode_count)

                    v_l = losses['vl']
                    p_l = losses['pl']
                    e_l = losses['ent']
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))

                    lr = sess.run([self.learning_rate])[0]
                    summary.value.add(tag='lr/lr', simple_value=float(lr))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                episode_count += 1
                if episode_count % 20000 == 0 and self.max_ep < 100:
                    self.max_ep += 10
                    

                sys.stdout.flush()

     
if __name__ == '__main__':           
    tf.reset_default_graph()
    
    env = test_prob.simul()
    link_size, node_size = env.get_num_params()
    a_size = link_size + node_size
    
    scale = 5
    action_width, action_center = env.get_link_scale(scale=scale)    
    
    total_episodes = 1e5#Set total number of episodes to train agent.
    savedir = 'log'

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    max_task_num = 4900
    interval = 0.1
    numpoints = int(scale/interval)
    dummy_y_house = np.zeros((max_task_num, numpoints, numpoints))

    taskseed = 0
    tasknum = 0
    z = np.indices((numpoints,numpoints)).transpose(1,2,0).reshape(-1,2)
    #gp = GaussianProcessRegressor(kernel=RBF(length_scale=0.1))
    #while tasknum < max_task_num:
    #    if tasknum % 100 == 0:
    #        print(tasknum,'/',max_task_num,'making loss surfaces...')
    #    np.random.seed(taskseed)
    #    length_scale = (np.random.uniform() * 0.6) + 1e-2
    #    gp = GaussianProcessRegressor(kernel=RBF(length_scale=length_scale))
    #    try:
    #        dummy_y = gp.sample_y(z/numpoints, random_state=tasknum)
    #    except:
    #        taskseed += 1
    #        continue
    #    dummy_y /= np.max(dummy_y)
    #    dummy_y_house[tasknum] = dummy_y.reshape(numpoints, numpoints)
    #    tasknum += 1
    #    taskseed += 1
    with open('scalerand_wh50_5000.p', 'rb') as f:
        _dummy_y_house = pickle.load(f)
    for di in range(_dummy_y_house.shape[0]):
        minval = np.min(_dummy_y_house[di])
        maxval = np.max(_dummy_y_house[di])
        _dummy_y_house[di] = (_dummy_y_house[di] - minval) / (maxval - minval) * 2 - 1
    dummy_y_house = _dummy_y_house[:max_task_num]

    test_y = _dummy_y_house[-100:]
    
    parameter_dict = {
            'a_size': a_size,
            'h_size': 48,
            'start_lr': 7e-4,
            'use_update_noise': False,
            'noise_start_scale': 0.01,
            'distance_threshold': 0.3,
            'gamma': 0.9,
            'max_ep': 10,
            'model_path': savedir,
            'use_noisynet': False,
            'dummy_y_house': dummy_y_house,
            'test_y': test_y
            }
    assert(not (parameter_dict['use_update_noise'] and parameter_dict['use_noisynet']))
    
    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False)
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        
        master_network = AC_Network(parameter_dict=parameter_dict, action_width=action_width, scope='global', trainer=None, global_step=global_step) # Generate global network
    #    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
        num_workers = 64 # Set workers to number of available CPU threads
        workers = []
        work_lock = Lock()
        shared_a = Array('d',int(a_size))
        #shared_a[:] = np.array(shared_a[:]) + 0.01
        shared_a[:] = np.random.rand(int(a_size))
        shared_r = Value('d',-np.inf)
        barrier = Barrier(num_workers)
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(i, parameter_dict, savedir, global_episodes, global_step, work_lock, shared_a, shared_r, barrier))
        saver = tf.train.Saver()
    
    
    seconfig = tf.ConfigProto(allow_soft_placement = True, intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    seconfig.gpu_options.allow_growth = True
    # Launch the tensorflow graph
    with tf.Session(config=seconfig) as sess:
        coord = tf.train.Coordinator()
        
        sess.run(tf.global_variables_initializer())
    
    #    s_init = np.random.rand(1,1,a_size)
        s_init = None
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(parameter_dict['gamma'], sess, coord, saver, total_episodes, s_init=s_init)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
            
        coord.join(worker_threads)

