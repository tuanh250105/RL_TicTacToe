import random
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from TicTacToeEnv import TicTacToeEnv

class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.1, board_size=20):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.board_size = board_size
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.results = {'wins': [], 'losses': [], 'draws': []}

    def state_to_key(self, state):
        return str(state.flatten().tobytes())

    def get_q_value(self, state, action):
        return self.q_table[self.state_to_key(state)][action]

    def choose_action(self, state, possible_actions):
        if not possible_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        max_q = float('-inf')
        best_actions = []
        for action in possible_actions:
            q_value = self.get_q_value(state, action)
            if q_value > max_q:
                max_q = q_value
                best_actions = [action]
            elif q_value == max_q:
                best_actions.append(action)
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, possible_next_actions):
        max_next_q = 0
        if possible_next_actions:
            max_next_q = max([self.get_q_value(next_state, next_action) for next_action in possible_next_actions], default=0)
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[self.state_to_key(state)][action] = new_q

    def train(self, episodes=5000, status_queue=None):
        env = TicTacToeEnv(board_size=self.board_size)
        for episode in range(episodes):
            state = env.reset()
            done = False
            wins, losses, draws = 0, 0, 0
            while not done:
                possible_actions = env.get_valid_actions()
                if not possible_actions:
                    break
                action = self.choose_action(state, possible_actions)
                next_state, reward, done = env.step(action, 1)
                if done:
                    if env.winner == 1:
                        reward = 1.0
                        wins += 1
                    elif env.winner == -1:
                        reward = -1.0
                        losses += 1
                    else:
                        reward = 0.5
                        draws += 1
                else:
                    possible_actions = env.get_valid_actions()
                    opponent_action = random.choice(possible_actions)
                    next_state, _, done = env.step(opponent_action, -1)
                    if done and env.winner == -1:
                        reward = -1.0
                        losses += 1
                next_possible_actions = env.get_valid_actions()
                self.update_q_table(state, action, reward, next_state, next_possible_actions)
                state = next_state
            self.results['wins'].append(wins)
            self.results['losses'].append(losses)
            self.results['draws'].append(draws)
            if status_queue and (episode + 1) % 500 == 0:
                progress = (episode + 1) / episodes * 100
                status_queue.put(f"progress:{progress:.1f}:Episode {episode + 1}/{episodes}, Epsilon: {self.epsilon:.4f}")

    def play(self, state):
        possible_actions = [(i, j) for i in range(self.board_size) for j in range(self.board_size) if state[i, j] == 0]
        old_epsilon = self.epsilon
        self.epsilon = 0
        action = self.choose_action(state, possible_actions)
        self.epsilon = old_epsilon
        return action

    def save_q_table(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))
            print(f"Loaded Q-table from {file_path}. States: {len(self.q_table)}")
            return True
        except FileNotFoundError:
            print(f"File {file_path} not found. Creating new Q-table.")
            return False

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['wins'], label='Wins')
        plt.plot(self.results['losses'], label='Losses')
        plt.plot(self.results['draws'], label='Draws')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        plt.title('Q-Learning Training Results (Tic-tac-toe)')
        plt.legend()
        plt.savefig(f'qlearning_tictactoe_{self.board_size}x{self.board_size}_results.png')
        plt.close()