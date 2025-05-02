import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
import matplotlib.pyplot as plt
from TicTacToeEnv import TicTacToeEnv
import json
class DQNAgent:
    """Tác nhân Deep Q-Learning cho trò chơi Tic-Tac-Toe."""
    
    def __init__(self, board_size=None, learning_rate=None, gamma=None, epsilon=None):
        """
        Khởi tạo tác nhân DQN.
        
        Args:
            board_size (int): Kích thước bàn cờ.
            learning_rate (float): Tốc độ học.
            gamma (float): Hệ số giảm giá trị tương lai.
            epsilon (float): Tỷ lệ khám phá.
        """
        with open('config.json', 'r') as f:
            config = json.load(f)
        dqn_config = config['dqn']
        self.board_size = board_size or config['board_size']
        self.state_size = (self.board_size, self.board_size, 1)
        self.action_size = self.board_size * self.board_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma or dqn_config['gamma']
        self.epsilon = epsilon or dqn_config['epsilon']
        self.epsilon_min = dqn_config['epsilon_min']
        self.epsilon_decay = dqn_config['epsilon_decay']
        self.learning_rate = learning_rate or dqn_config['learning_rate']
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
        """Xây dựng mô hình CNN cho DQN."""
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
        """Chuyển trạng thái thành đầu vào cho mô hình."""
        return np.reshape(state, (1, self.board_size, self.board_size, 1))

    def action_to_coord(self, action):
        """Chuyển chỉ số hành động thành tọa độ (i, j)."""
        return divmod(action, self.board_size)

    def choose_action(self, state, valid_actions):
        """Chọn hành động theo chiến lược epsilon-greedy."""
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        state_input = self.state_to_input(state)
        act_values = self.model.predict(state_input, verbose=0)[0]
        valid_indices = [i * self.board_size + j for i, j in valid_actions]
        valid_q_values = act_values[valid_indices]
        max_idx = np.argmax(valid_q_values)
        return valid_actions[max_idx]

    def remember(self, state, action, reward, next_state, done, next_valid_actions):
        """Lưu kinh nghiệm vào bộ nhớ replay."""
        self.memory.append((state, action, reward, next_state, done, next_valid_actions))

    def replay(self, batch_size):
        """Huấn luyện mô hình bằng kinh nghiệm replay."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done, next_valid_actions in minibatch:
            state_input = np.reshape(state, (self.board_size, self.board_size, 1))
            states.append(state_input)
            target = reward
            if not done:
                next_state_input = self.state_to_input(next_state)
                next_q_values = self.target_model.predict(next_state_input, verbose=0)[0]
                valid_indices = [i * self.board_size + j for i, j in next_valid_actions]
                valid_q_values = next_q_values[valid_indices]
                target = reward + self.gamma * np.max(valid_q_values) if valid_indices else reward
            action_idx = action[0] * self.board_size + action[1]
            target_f = self.model.predict(self.state_to_input(state), verbose=0)[0]
            target_f[action_idx] = target
            targets.append(target_f)
        states = np.array(states)
        targets = np.array(targets)
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.update_target_counter += 1
        if self.update_target_counter >= self.update_target_freq:
            self.target_model.set_weights(self.model.get_weights())
            self.update_target_counter = 0

    def heuristic_opponent_action(self, possible_actions):
        """Chọn nước đi của đối thủ, ưu tiên gần trung tâm."""
        center = self.board_size // 2
        scored_actions = [
            (abs(i - center) + abs(j - center), (i, j))
            for i, j in possible_actions
        ]
        scored_actions.sort()
        return scored_actions[0][1] if scored_actions else random.choice(possible_actions)

    def train(self, batch_size=64, episodes=2000, status_queue=None):
        """Huấn luyện tác nhân qua số lượng episodes."""
        with open('config.json', 'r') as f:
            config = json.load(f)
        batch_size = batch_size or config['dqn']['batch_size']
        episodes = episodes or config['dqn']['episodes']
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
                    opponent_action = self.heuristic_opponent_action(valid_actions)
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
        """Lưu mô hình DQN vào file."""
        with open('config.json', 'r') as f:
            config = json.load(f)
        file_path = file_path or config['dqn']['model_path']
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Tải mô hình DQN từ file."""
        with open('config.json', 'r') as f:
            config = json.load(f)
        file_path = file_path or config['dqn']['model_path']
        try:
            self.model = tf.keras.models.load_model(file_path)
            self.target_model = tf.keras.models.load_model(file_path)
            print(f"Loaded DQN model from {file_path}")
            return True
        except (OSError, ValueError) as e:
            print(f"Failed to load model from {file_path}: {e}")
            return False

    def plot_results(self):
        """Vẽ biểu đồ kết quả huấn luyện."""
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