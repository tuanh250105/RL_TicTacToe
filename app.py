from flask import Flask, render_template, request, jsonify
from TicTacToeEnv import TicTacToeEnv
from QLearningAgent import QLearningAgent
from DQNAgent import DQNAgent
import numpy as np
import json

app = Flask(__name__)

# Tắt cảnh báo TensorFlow oneDNN
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Khởi tạo biến toàn cục
with open('config.json', 'r') as f:
    config = json.load(f)
board_size = config['board_size']
agent_type = "qlearning"
env = TicTacToeEnv(board_size=board_size)
agent = None
stats = {'player_wins': 0, 'agent_wins': 0, 'draws': 0}

def init_agent():
    """Khởi tạo tác nhân dựa trên agent_type."""
    global agent, agent_type, board_size
    if agent_type == "qlearning":
        agent = QLearningAgent(board_size=board_size)
        agent.load_q_table(config['qlearning']['q_table_path'])
    else:
        agent = DQNAgent(board_size=board_size)
        agent.load_model(config['dqn']['model_path'])

init_agent()

@app.route('/')
def index():
    """Hiển thị giao diện trò chơi."""
    return render_template('index.html', board_size=board_size, stats=stats, agent_type=agent_type, epsilon=agent.epsilon)

@app.route('/move', methods=['POST'])
def make_move():
    """Xử lý nước đi của người chơi và tác nhân."""
    global env, stats
    data = request.json
    row, col = int(data['row']), int(data['col'])
    
    # Kiểm tra nước đi hợp lệ
    if env.board[row, col] != 0 or env.done:
        return jsonify({
            'board': env.board.tolist(),
            'done': env.done,
            'winner': env.winner,
            'stats': stats,
            'epsilon': agent.epsilon
        })
    
    # Nước đi của người chơi (X)
    state, reward, done = env.step((row, col), 1)
    if done:
        if env.winner == 1:
            stats['player_wins'] += 1
        elif env.winner == -1:
            stats['agent_wins'] += 1
        else:
            stats['draws'] += 1
        return jsonify({
            'board': state.tolist(),
            'done': done,
            'winner': env.winner,
            'stats': stats,
            'epsilon': agent.epsilon
        })
    
    # Nước đi của tác nhân (O)
    agent_action = agent.play(env.board)
    if agent_action:
        state, reward, done = env.step(agent_action, -1)
        if done:
            if env.winner == 1:
                stats['player_wins'] += 1
            elif env.winner == -1:
                stats['agent_wins'] += 1
            else:
                stats['draws'] += 1
    
    return jsonify({
        'board': state.tolist(),
        'done': done,
        'winner': env.winner,
        'stats': stats,
        'epsilon': agent.epsilon
    })

@app.route('/reset', methods=['POST'])
def reset_game():
    """Đặt lại trò chơi."""
    global env
    env = TicTacToeEnv(board_size=board_size)
    return jsonify({
        'board': env.board.tolist(),
        'done': False,
        'winner': None,
        'stats': stats,
        'epsilon': agent.epsilon
    })

@app.route('/set_config', methods=['POST'])
def set_config():
    """Cập nhật agent_type và board_size."""
    global board_size, agent_type, env
    data = request.json
    new_board_size = int(data['board_size'])
    new_agent_type = data['agent_type']
    
    # Cập nhật cấu hình
    if new_board_size != board_size or new_agent_type != agent_type:
        board_size = new_board_size
        agent_type = new_agent_type
        env = TicTacToeEnv(board_size=board_size)
        init_agent()
    
    return jsonify({
        'board': env.board.tolist(),
        'done': False,
        'winner': None,
        'stats': stats,
        'epsilon': agent.epsilon
    })

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    """Đặt lại thống kê về 0."""
    global stats
    stats = {'player_wins': 0, 'agent_wins': 0, 'draws': 0}
    return jsonify({
        'stats': stats,
        'epsilon': agent.epsilon
    })

if __name__ == '__main__':
    app.run(debug=True)