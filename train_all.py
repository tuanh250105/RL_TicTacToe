from QLearningAgent import QLearningAgent
from DQNAgent import DQNAgent
import json

# Đọc cấu hình
with open('config.json', 'r') as f:
    config = json.load(f)
q_config = config['qlearning']
dqn_config = config['dqn']
board_size = config['board_size']

# Huấn luyện Q-Learning
print("Starting Q-Learning training...")
q_agent = QLearningAgent(
    alpha=q_config['alpha'],
    gamma=q_config['gamma'],
    epsilon=q_config['epsilon'],
    board_size=board_size
)
q_agent.train(episodes=q_config['episodes'])
q_agent.save_q_table(q_config['q_table_path'])
q_agent.plot_results()
print("Q-Learning training completed. Q-table saved to", q_config['q_table_path'])

# Huấn luyện DQN
print("Starting DQN training...")
dqn_agent = DQNAgent(
    board_size=board_size,
    learning_rate=dqn_config['learning_rate'],
    gamma=dqn_config['gamma'],
    epsilon=dqn_config['epsilon']
)
dqn_agent.train(
    batch_size=dqn_config['batch_size'],
    episodes=dqn_config['episodes']
)
dqn_agent.save_model(dqn_config['model_path'])
dqn_agent.plot_results()
print("DQN training completed. Model saved to", dqn_config['model_path'])