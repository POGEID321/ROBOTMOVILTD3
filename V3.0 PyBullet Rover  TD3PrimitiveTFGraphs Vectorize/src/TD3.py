import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.optimizers import Adam
import pickle
import os
from keras.losses import MeanSquaredError
import time

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=0.99, update_actor_interval=2, warmup=1000,
                 n_actions=2, max_size=1000000, layer1_size=500,
                 layer2_size=400, layer3_size=400, batch_size=100, 
                 noise=0.1):
        
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space_high
        self.min_action = env.action_space_low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.q_value = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.critic_1_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.critic_2_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.actor_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        self.mse_loss = tf.keras.losses.MeanSquaredError()

        self.actor = ActorNetwork(layer1_size, layer2_size, layer3_size,
                                  n_actions=n_actions, name='actor')

        self.critic_1 = CriticNetwork(layer1_size, layer2_size, layer3_size,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(layer1_size, layer2_size, layer3_size,
                                      name='critic_2')

        self.target_actor = ActorNetwork(layer1_size, layer2_size, layer3_size,
                                         n_actions=n_actions,
                                         name='target_actor')
        self.target_critic_1 = CriticNetwork(layer1_size, layer2_size, layer3_size,
                                             name='target_critic_1')
        self.target_critic_2 = CriticNetwork(layer1_size, layer2_size, layer3_size,
                                             name='target_critic_2')

        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        
        self.critic_1.compile(optimizer=Adam(learning_rate=beta),
                              loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=beta),
                              loss='mean_squared_error')

        self.target_actor.compile(optimizer=Adam(learning_rate=alpha),
                                  loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta),
                                     loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta),
                                     loss='mean_squared_error')

        self.noise = noise
        self.update_network_parameters(tau=1)

        self.ckpt = tf.train.Checkpoint(
            actor=self.actor,
            actor_target=self.target_actor,
            critic_1=self.critic_1,
            critic_2=self.critic_2,
            critic_target_1=self.target_critic_1,
            critic_target_2=self.target_critic_2,
            actor_optimizer=self.actor.optimizer,
            critic_optimizer_1=self.critic_1.optimizer,
            critic_optimizer_2=self.critic_2.optimizer,
            step=tf.Variable(0)  # opcional: para llevar el conteo del paso
        )

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory='tmp/td3/checkpoints', max_to_keep=5)


    def Ajuste_Ruido(self):
        self.noise=self.noise*0.9
        self.noise = max(self.noise, 0.01) 

    def choose_action(self, observation):   #ecuacion clave suavizado
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            # returns a batch size of 1, want a scalar array
            mu = self.actor(state)[0]
        mu_prime = mu + np.random.normal(scale=self.noise)

        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        return mu_prime

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return
        self.Graph_Learn()

    @tf.function
    def Graph_Learn(self):

        states, actions, rewards, new_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:  #Ecuacion de Valor
            target_actions = self.target_actor(states_)
            target_actions = target_actions + \
                tf.clip_by_value(tf.random.normal(shape=target_actions.shape, stddev=0.2), -0.5, 0.5)

            target_actions = tf.clip_by_value(target_actions, self.min_action,
                                              self.max_action)

            q1_ = self.target_critic_1(states_, target_actions)
            q2_ = self.target_critic_2(states_, target_actions)

            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            # shape is [batch_size, 1], want to collapse to [batch_size]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)
            self.q_value.assign(tf.reduce_mean((q1 + q2) / 2.0))
            critic_value_ = tf.math.minimum(q1_, q2_)
            # in tf2 only integer scalar arrays can be used as indices
            # and eager exection doesn't support assignment, so we can't do
            # q1_[dones] = 0.0
            target = rewards + self.gamma*critic_value_*(1-dones) #ECUACION IMPLICITA DE VALOR

            #ECUACION DE PERDIDA
            critic_1_loss = self.mse_loss(target, q1)
            critic_2_loss = self.mse_loss(target, q2)
            self.critic_1_loss.assign(critic_1_loss)
            self.critic_2_loss.assign(critic_2_loss)



        critic_1_gradient = tape.gradient(critic_1_loss,self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss,self.critic_2.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.q_value.assign(tf.reduce_mean((q1 + q2) / 2.0))
        self.learn_step_cntr.assign_add(1)

        if self.learn_step_cntr % self.update_actor_iter == 0:
            self._update_actor(states)

        self.update_network_parameters()

    @tf.function
    def _update_actor(self, states):
        with tf.GradientTape() as tape: 
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss,self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
        self.actor_loss.assign(actor_loss)

    @tf.function
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Actor
        for target_var, main_var in zip(self.target_actor.variables, self.actor.variables):
            target_var.assign(tau * main_var + (1.0 - tau) * target_var)

        # Critic 1
        for target_var, main_var in zip(self.target_critic_1.variables, self.critic_1.variables):
            target_var.assign(tau * main_var + (1.0 - tau) * target_var)

        # Critic 2
        for target_var, main_var in zip(self.target_critic_2.variables, self.critic_2.variables):
            target_var.assign(tau * main_var + (1.0 - tau) * target_var)

    # def update_network_parameters(self, tau=None):
    #     if tau is None:
    #         tau = self.tau

    #     weights = []
    #     targets = self.target_actor.weights
    #     for i, weight in enumerate(self.actor.weights):
    #         weights.append(weight * tau + targets[i]*(1-tau))

    #     self.target_actor.set_weights(weights)

    #     weights = []
    #     targets = self.target_critic_1.weights
    #     for i, weight in enumerate(self.critic_1.weights):
    #         weights.append(weight * tau + targets[i]*(1-tau))

    #     self.target_critic_1.set_weights(weights)

    #     weights = []
    #     targets = self.target_critic_2.weights
    #     for i, weight in enumerate(self.critic_2.weights):
    #         weights.append(weight * tau + targets[i]*(1-tau))

    #     self.target_critic_2.set_weights(weights)

    def save_models(self):

        print('... saving models ...')
        self.ckpt.step.assign(self.learn_step_cntr)
        self.ckpt_manager.save()
        print('Modelos guardados.')

        # ✅ Guardar el replay buffer
        with open('replay_buffer.pkl', 'wb') as f:
            pickle.dump(self.memory, f)

        # dummy_state = tf.random.uniform((1, 7), dtype=tf.float32)
        # dummy_action = tf.random.uniform((1, 2), dtype=tf.float32)

        # _ = self.actor(dummy_state)
        # self.actor.save("actor_saved_model", save_format="tf")

        # _ = self.critic_1(dummy_state, dummy_action)
        # self.critic_1.save("critic1_saved_model", save_format="tf")

        # _ = self.critic_2(dummy_state, dummy_action)
        # self.critic_2.save("critic2_saved_model", save_format="tf")
        
        print(self.memory.mem_cntr)
        print(self.learn_step_cntr)
        print(self.memory.state_memory)

    def load_models(self):

        print('... loading models ...')
        latest_ckpt = self.ckpt_manager.latest_checkpoint

        if latest_ckpt:
            self.ckpt.restore(latest_ckpt)
            self.learn_step_cntr.assign(self.ckpt.step)
            print(f'Modelos restaurados desde {latest_ckpt}')
            # ✅ Cargar replay buffer si existe
            if os.path.exists('replay_buffer.pkl'):
                with open('replay_buffer.pkl', 'rb') as f:
                    self.memory = pickle.load(f)
                print('Replay buffer restaurado.')
            else:
                print('No se encontró replay buffer.')
        else:
            print('No se encontró checkpoint.')

        print(self.memory.mem_cntr)
        print(self.learn_step_cntr)
        print(self.memory.state_memory)

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, fc3_dims, n_actions, name,chkpt_dir="tmp\\td3"):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_td3.weights.keras")

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc3_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)

        return mu

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, fc3_dims, name, chkpt_dir='tmp\\td3'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3.weights.keras')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc3_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        q1_action_value = self.fc1(tf.concat([state, action], axis=1))
        q1_action_value = self.fc2(q1_action_value)

        q = self.q(q1_action_value)

        return q

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):   
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

