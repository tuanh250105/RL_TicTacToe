
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import matplotlib.pyplot as plt
from TicTacToeEnv import TicTacToeEnv

class DQNAgent:
    def __init__(self, board_size=20, learning_rate=0.001, gamma=0.95, epsilon=1.0):
        self.board_size = board_size
        self.state_size = (self.board_size, self.board_size, 1)  # Shape for CNN
        self.action_size = self.board_size * self.board_size  # Possible actions
        self.memory = deque(maxlen=5000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        print(f"Initializing DQNAgent with state_size: {self.state_size}")
        try:
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.target_model.set_weights(self.model.get_weights())
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
        self.results = {'wins': [], 'losses': [], 'draws': []}
        self.update_target_counter = 0
        self.update_target_freq = 100

    def build_model(self):
        print(f"Building model with input shape: {self.state_size}")
        model = tf.keras.Sequential([
            tf.keras.Input(shape=self.state_size),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def state_to_input(self, state):
        return np.reshape(state, (1, self.board_size, self.board_size, 1))

    def action_to_coord(self, action):
        return divmod(action, self.board_size)

    def choose_action(self, state, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        state_input = self.state_to_input(state)
        act_values = self.model.predict(state_input, verbose=0)[0]
        masked_values = np.full(self.action_size, -np.inf)
        for i, j in valid_actions:
            action_idx = i * self.board_size + j
            masked_values[action_idx] = act_values[action_idx]
        return self.action_to_coord(np.argmax(masked_values))

    def remember(self, state, action, reward, next_state, done, next_valid_actions):
        self.memory.append((state, action, reward, next_state, done, next_valid_actions))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done, next_valid_actions in minibatch:
            target = reward
            if not done:
                next_state_input = self.state_to_input(next_state)
                next_q_values = self.target_model.predict(next_state_input, verbose=0)[0]
                valid_q_values = [next_q_values[i * self.board_size + j] for i, j in next_valid_actions]
                target = reward + self.gamma * np.max(valid_q_values) if valid_q_values else reward
            state_input = self.state_to_input(state)
            action_idx = action[0] * self.board_size + action[1]
            target_f = self.model.predict(state_input, verbose=0)
            target_f[0][action_idx] = target
            self.model.fit(state_input, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_target_counter += 1
        if self.update_target_counter >= self.update_target_freq:
            self.target_model.set_weights(self.model.get_weights())
            self.update_target_counter = 0

    def train(self, batch_size=64, episodes=2000, status_queue=None):
        env = TicTacToeEnv(board_size=self.board_size)
        for e in range(episodes):
            state = env.reset()
            done = False
            wins, losses, draws = 0, 0, 0
            while not done:
                valid_actions = env.get_valid_actions()
                action = self.choose_action(state, valid_actions)
                next_state, reward, done = env.step(action, 1)
                if done:
                    if env.winner == 1:
                        wins += 1
                    elif env.winner == -1:
                        losses += 1
                    else:
                        draws += 1
                else:
                    valid_actions = env.get_valid_actions()
                    opponent_action = random.choice(valid_actions)
                    next_state, _, done = env.step(opponent_action, -1)
                    if done and env.winner == -1:
                        reward = -1
                        losses += 1
                next_valid_actions = env.get_valid_actions()
                self.remember(state, action, reward, next_state, done, next_valid_actions)
                state = next_state
                self.replay(batch_size)
            self.results['wins'].append(wins)
            self.results['losses'].append(losses)
            self.results['draws'].append(draws)
            if status_queue and (e + 1) % 100 == 0:
                progress = (e + 1) / episodes * 100
                status_queue.put(f"progress:{progress:.1f}:Episode {e + 1}/{episodes}, Epsilon: {self.epsilon:.2f}")

    def save_model(self, file_path):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        try:
            self.model = tf.keras.models.load_model(file_path)
            self.target_model = tf.keras.models.load_model(file_path)
            print(f"Loaded DQN model from {file_path}")
            return True
        except Exception as e:
            print(f"Failed to load model from {file_path}: {e}")
            return False

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['wins'], label='Wins')
        plt.plot(self.results['losses'], label='Losses')
        plt.plot(self.results['draws'], label='Draws')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        plt.title('DQN Training Results (Tic-tac-toe)')
        plt.legend()
        plt.savefig(f'dqn_tictactoe_{self.board_size}x{self.board_size}_results.png')
        plt.close()