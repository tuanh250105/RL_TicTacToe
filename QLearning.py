import turtle
import random
import time
import numpy as np
import pickle
from tkinter import Tk, Button, Label, Frame
from tkinter import messagebox

class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.1):
        """
        Khởi tạo tham số cho Q-learning
        alpha: tỉ lệ học (learning rate)
        gamma: hệ số chiết khấu (discount factor)
        epsilon: tỉ lệ thăm dò (exploration rate)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Q-table lưu dạng dictionary
        
    def get_q_value(self, state, action):
        """Lấy giá trị Q cho cặp state-action"""
        # Chuyển state thành tuple để có thể hash làm key trong dictionary
        state_tuple = tuple(map(tuple, state))
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {}
        if action not in self.q_table[state_tuple]:
            self.q_table[state_tuple][action] = 0.0
        return self.q_table[state_tuple][action]
    
    def choose_action(self, state, possible_actions):
        """
        Chọn hành động theo chính sách ɛ-greedy
        state: trạng thái hiện tại của bàn cờ
        possible_actions: danh sách các hành động có thể
        """
        if not possible_actions:
            return None
            
        # Chuyển thành tuple để dùng làm key
        state_tuple = tuple(map(tuple, state))
        
        # Epsilon-greedy: với xác suất epsilon, chọn random để khám phá
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        # Tìm nước đi tốt nhất theo giá trị Q
        max_q = float('-inf')
        best_actions = []
        
        for action in possible_actions:
            q_value = self.get_q_value(state, action)
            if q_value > max_q:
                max_q = q_value
                best_actions = [action]
            elif q_value == max_q:
                best_actions.append(action)
        
        # Nếu có nhiều nước tốt như nhau, chọn ngẫu nhiên một trong số đó
        return random.choice(best_actions)
    
    def update_q_table(self, state, action, reward, next_state, possible_next_actions):
        """
        Cập nhật giá trị Q-table theo công thức Bellman
        """
        # Tính giá trị Q tối đa cho trạng thái tiếp theo
        max_next_q = 0
        if possible_next_actions:  # Nếu còn nước đi tiếp theo
            max_next_q = max([self.get_q_value(next_state, next_action) for next_action in possible_next_actions], default=0)
        
        # Cập nhật giá trị Q theo công thức Q-learning
        current_q = self.get_q_value(state, action)
        state_tuple = tuple(map(tuple, state))
        
        # Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_tuple][action] = new_q
    
    def train(self, episodes=10000):
        """
        Huấn luyện agent trong một số trận đấu
        """
        for episode in range(episodes):
            board = make_empty_board(3)  # Bàn cờ 3x3 cho Tic Tac Toe
            game_over = False
            
            # AI chơi với chính nó để học
            while not game_over:
                # Lượt của agent X
                state_x = [row[:] for row in board]
                possible_actions_x = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']
                
                if not possible_actions_x:  # Hòa
                    game_over = True
                    continue
                
                action_x = self.choose_action(state_x, possible_actions_x)
                board[action_x[0]][action_x[1]] = 'X'
                
                # Kiểm tra kết thúc trò chơi
                result = is_win(board)
                if result != 'Continue playing':
                    if result == 'X won':
                        self.update_q_table(state_x, action_x, 1.0, board, [])  # Thắng: thưởng 1.0
                    elif result == 'Draw':
                        self.update_q_table(state_x, action_x, 0.5, board, [])  # Hòa: thưởng 0.5
                    game_over = True
                    continue
                
                # Lượt của agent O
                state_o = [row[:] for row in board]
                possible_actions_o = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']
                
                if not possible_actions_o:  # Hòa
                    game_over = True
                    continue
                    
                action_o = self.choose_action(state_o, possible_actions_o)
                board[action_o[0]][action_o[1]] = 'O'
                
                # Kiểm tra kết thúc trò chơi
                result = is_win(board)
                if result != 'Continue playing':
                    if result == 'O won':
                        self.update_q_table(state_o, action_o, 1.0, board, [])  # Thắng: thưởng 1.0
                        # Cập nhật X là thua
                        self.update_q_table(state_x, action_x, -1.0, state_o, possible_actions_o)
                    elif result == 'Draw':
                        self.update_q_table(state_o, action_o, 0.5, board, [])  # Hòa: thưởng 0.5
                        self.update_q_table(state_x, action_x, 0.5, state_o, possible_actions_o)
                    game_over = True
                else:
                    # Cập nhật X dựa trên trạng thái mới
                    next_possible_actions_x = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']
                    self.update_q_table(state_x, action_x, 0.0, board, next_possible_actions_x)
                    
            # Giảm epsilon theo thời gian (giảm dần việc khám phá ngẫu nhiên)
            if (episode + 1) % 1000 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.95)
                print(f"Đã hoàn thành {episode + 1} episodes. Epsilon: {self.epsilon:.4f}")
    
    def play(self, board):
        """
        Trả về nước đi tốt nhất dựa trên Q-table đã học
        """
        possible_actions = [(i, j) for i in range(len(board)) for j in range(len(board)) if board[i][j] == ' ']
        
        # Trong quá trình chơi thực, sử dụng epsilon=0 để luôn chọn nước đi tốt nhất
        old_epsilon = self.epsilon
        self.epsilon = 0
        action = self.choose_action(board, possible_actions)
        self.epsilon = old_epsilon
        
        return action
    
    def save_q_table(self, file_path):
        """Lưu Q-table vào file"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self, file_path):
        """Tải Q-table từ file"""
        try:
            with open(file_path, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Đã tải Q-table từ {file_path}. Số lượng trạng thái: {len(self.q_table)}")
            return True
        except FileNotFoundError:
            print(f"Không tìm thấy file {file_path}. Sẽ tạo Q-table mới.")
            return False

# Các hàm từ mã nguồn gốc, được điều chỉnh cho Tic Tac Toe
def make_empty_board(sz):
    board = []
    for i in range(sz):
        board.append([" "]*sz)
    return board

def is_empty(board):
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] != ' ':
                return False
    return True

def is_in(board, y, x):
    return 0 <= y < len(board) and 0 <= x < len(board)

def is_win(board):
    """Kiểm tra người thắng trên bàn cờ Tic Tac Toe"""
    # Kiểm tra hàng ngang
    for i in range(len(board)):
        if board[i][0] != ' ' and board[i][0] == board[i][1] == board[i][2]:
            return f"{board[i][0]} won"
    
    # Kiểm tra hàng dọc
    for i in range(len(board)):
        if board[0][i] != ' ' and board[0][i] == board[1][i] == board[2][i]:
            return f"{board[0][i]} won"
    
    # Kiểm tra đường chéo chính
    if board[0][0] != ' ' and board[0][0] == board[1][1] == board[2][2]:
        return f"{board[0][0]} won"
    
    # Kiểm tra đường chéo phụ
    if board[0][2] != ' ' and board[0][2] == board[1][1] == board[2][0]:
        return f"{board[0][2]} won"
    
    # Kiểm tra hòa
    if all(board[i][j] != ' ' for i in range(len(board)) for j in range(len(board))):
        return 'Draw'
    
    return 'Continue playing'

# Thiết lập giao diện đồ họa và xử lý sự kiện
global move_history, board, screen, colors, win, mode, current_turn, agent

def restart():
    """Restart the board and start a new game."""
    global board, win, move_history
    move_history = []
    win = False
    board = make_empty_board(len(board))
    screen.clearscreen()
    initialize_game()

def select_mode():
    """Exit the current game and return to the mode selection screen."""
    global root
    root.withdraw()  # Hide the tkinter window during mode selection
    screen.bye()
    initialize()

def initialize_tkinter():
    root = Tk()
    root.title("Tic Tac Toe Q-Learning")

    frame = Frame(root)
    frame.pack(pady=10)

    restart_button = Button(frame, text="Restart", command=restart, width=15, height=2)
    restart_button.grid(row=0, column=0, padx=10, pady=10)

    select_mode_button = Button(frame, text="Select Mode", command=select_mode, width=15, height=2)
    select_mode_button.grid(row=0, column=1, padx=10, pady=10)

    train_button = Button(frame, text="Train AI", command=train_ai, width=15, height=2)
    train_button.grid(row=1, column=0, padx=10, pady=10)

    save_button = Button(frame, text="Save AI", command=save_ai, width=15, height=2)
    save_button.grid(row=1, column=1, padx=10, pady=10)

    status_label = Label(root, text="Ready", bd=1, relief='sunken', anchor='w')
    status_label.pack(side='bottom', fill='x')

    root.geometry("350x200")
    root.withdraw()
    return root, status_label

def click(x, y):
    global board, colors, win, move_history, mode, current_turn, agent
    
    if win:
        return
    
    grid_x, grid_y = getindexposition(x, y)
    
    # Trả lại nếu click ngoài bàn cờ
    if not is_in(board, grid_y, grid_x):
        return

    if board[grid_y][grid_x] == ' ' and not win:
        if mode == 1:  # Người vs Người
            # Lượt của người chơi hiện tại
            marker = 'X' if current_turn == 'X' else 'O'
            color_key = 'b' if marker == 'X' else 'w'
            
            draw_stone(grid_x, grid_y, colors[color_key])
            board[grid_y][grid_x] = marker
            move_history.append((grid_x, grid_y))
            
            game_res = is_win(board)
            if game_res in ["X won", "O won", "Draw"]:
                print(game_res)
                messagebox.showinfo('Kết quả', game_res)
                win = True
                return
            
            # Chuyển lượt
            current_turn = 'O' if current_turn == 'X' else 'X'

        elif mode == 2:  # Người vs AI
            # Lượt của người chơi (X)
            draw_stone(grid_x, grid_y, colors['b'])
            board[grid_y][grid_x] = 'X'
            move_history.append((grid_x, grid_y))
            
            game_res = is_win(board)
            if game_res in ["X won", "O won", "Draw"]:
                print(game_res)
                messagebox.showinfo('Kết quả', game_res)
                win = True
                return

            # Lượt của AI (O)
            ai_move = agent.play(board)
            if ai_move:
                ay, ax = ai_move
                draw_stone(ax, ay, colors['w'])
                board[ay][ax] = 'O'
                move_history.append((ax, ay))
                
                game_res = is_win(board)
                if game_res in ["X won", "O won", "Draw"]:
                    print(game_res)
                    messagebox.showinfo('Kết quả', game_res)
                    win = True
                    return

def train_ai():
    """Train the AI agent."""
    global agent, status_label
    
    status_label.config(text="Đang huấn luyện AI...")
    root.update()
    
    # Huấn luyện trong một thread riêng để không làm đóng băng giao diện
    import threading
    
    def training_thread():
        agent.train(episodes=5000)
        status_label.config(text="Huấn luyện AI hoàn tất")
    
    thread = threading.Thread(target=training_thread)
    thread.daemon = True  # Thread sẽ tự động kết thúc khi chương trình chính kết thúc
    thread.start()

def save_ai():
    """Save the trained AI."""
    global agent, status_label
    
    try:
        agent.save_q_table("tictactoe_qtable.pkl")
        status_label.config(text="Đã lưu AI thành công!")
    except Exception as e:
        status_label.config(text=f"Lỗi khi lưu AI: {e}")

def initialize_game():
    global win, board, screen, colors, root, status_label
    
    move_history = []
    win = False
    
    # Khởi tạo bàn cờ 3x3 cho Tic Tac Toe
    board = make_empty_board(3)
    
    screen = turtle.Screen()
    screen.onclick(click)
    screen.setup(400, 400)
    screen.setworldcoordinates(-1, 3, 3, -1)
    screen.bgcolor('orange')
    screen.tracer(500)

    colors = {'w': turtle.Turtle(), 'b': turtle.Turtle(), 'g': turtle.Turtle()}
    colors['w'].color('white')  # O
    colors['b'].color('black')  # X

    for key in colors:
        colors[key].ht()
        colors[key].penup()
        colors[key].speed(0)

    # Vẽ lưới
    border = turtle.Turtle()
    border.speed(0)
    border.penup()

    # Vẽ lưới ngang
    for i in range(4):
        border.penup()
        border.goto(0, i)
        border.pendown()
        border.goto(3, i)
    
    # Vẽ lưới dọc
    for i in range(4):
        border.penup()
        border.goto(i, 0)
        border.pendown()
        border.goto(i, 3)

    border.ht()
    root.deiconify()  # Hiển thị lại cửa sổ tkinter
    screen.listen()

def initialize():
    global win, board, screen, colors, move_history, mode, current_turn, agent, root, status_label

    move_history = []
    win = False

    # Khởi tạo agent Q-Learning
    agent = QLearningAgent(alpha=0.3, gamma=0.9, epsilon=0.1)
    # Thử tải Q-table đã lưu trước đó
    agent.load_q_table("tictactoe_qtable.pkl")

    # Lựa chọn chế độ chơi
    mode = None
    while mode not in [1, 2]:
        try:
            mode = int(turtle.textinput("Chọn chế độ chơi", "1: Người vs Người\n2: Người vs AI\nNhập lựa chọn:"))
        except:
            continue

    if mode == 1:
        current_turn = 'X'
    elif mode == 2:
        current_turn = None

    root, status_label = initialize_tkinter()
    initialize_game()
    root.mainloop()
    
def getindexposition(x, y):
    """Chuyển đổi vị trí click chuột thành tọa độ lưới"""
    intx, inty = int(x), int(y)
    return intx, inty

def draw_stone(x, y, colturtle):
    """Vẽ X hoặc O trên bàn cờ"""
    colturtle.penup()
    colturtle.goto(x + 0.5, y + 0.5)
    
    # Xác định marker dựa vào màu
    if colturtle == colors['b']:  # X (đen)
        colturtle.pendown()
        colturtle.goto(x + 0.2, y + 0.2)
        colturtle.goto(x + 0.8, y + 0.8)
        colturtle.penup()
        colturtle.goto(x + 0.8, y + 0.2)
        colturtle.pendown()
        colturtle.goto(x + 0.2, y + 0.8)
        colturtle.penup()
    else:  # O (trắng)
        colturtle.goto(x + 0.5, y + 0.2)
        colturtle.pendown()
        colturtle.circle(0.3)
        colturtle.penup()
    
if __name__ == '__main__':
    initialize()