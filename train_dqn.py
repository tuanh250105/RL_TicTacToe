from DQNAgent import DQNAgent
import json

with open('config.json', 'r') as f:
    config = json.load(f)
dqn_config = config['dqn']
board_size = config['board_size']

agent = DQNAgent(
    board_size=board_size,
    learning_rate=dqn_config['learning_rate'],
    gamma=dqn_config['gamma'],
    epsilon=dqn_config['epsilon']
)
agent.train(
    batch_size=dqn_config['batch_size'],
    episodes=dqn_config['episodes']
)
agent.save_model(dqn_config['model_path'])
agent.plot_results()

print("DQN training completed. Model saved to", dqn_config['model_path'])