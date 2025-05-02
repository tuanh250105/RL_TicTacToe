import tkinter as tk
from tkinter import messagebox, ttk
import threading
import queue
import logging
import csv
import os
from TicTacToeEnv import TicTacToeEnv
from DQNAgent import DQNAgent
from QLearning import QLearningAgent

# Set up logging
logging.basicConfig(filename='tictactoe_ui.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TicTacToeUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe")
        self.board_size = 20  # Default board size
        self.move_history = []
        self.training_thread = None
        self.is_training = False
        logging.info("Initializing TicTacToeEnv")
        print("Initializing TicTacToeEnv")
        self.env = TicTacToeEnv(board_size=self.board_size)
        logging.info("Initializing DQNAgent")
        print("Initializing DQNAgent")
        try:
            self.dqn_agent = DQNAgent(board_size=self.board_size)
        except Exception as e:
            logging.error(f"Error initializing DQNAgent: {e}")
            print(f"Error initializing DQNAgent: {e}")
            messagebox.showerror("Error", f"Failed to initialize DQNAgent: {e}")
            raise
        logging.info("Initializing QLearningAgent")
        print("Initializing QLearningAgent")
        self.qlearning_agent = QLearningAgent(board_size=self.board_size)
        self.current_player = 1
        self.mode = None
        self.algorithm = None
        self.buttons = []
        self.status_queue = queue.Queue()
        logging.info("Setting up UI")
        print("Setting up UI")
        self.setup_ui()
        self.check_status_queue()
        logging.info("UI setup completed")
        print("UI setup completed")

    def setup_ui(self):
        self.root.geometry("950x750")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")

        self.header_frame = tk.Frame(self.root, bg="#0288D1")
        self.header_frame.pack(fill='x', pady=10)
        tk.Label(self.header_frame, text="Tic-Tac-Toe", font=("Arial", 24, "bold"), bg="#0288D1", fg="white").pack(pady=5)
        tk.Label(self.header_frame, text="Play against AI or another player!", font=("Arial", 12), bg="#0288D1", fg="white").pack()

        self.control_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.control_frame.pack(pady=10)

        # Board size selection
        size_frame = tk.Frame(self.control_frame, bg="#f0f0f0")
        size_frame.grid(row=0, column=0, columnspan=3, pady=5)
        tk.Label(size_frame, text="Board Size:", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(side='left', padx=5)
        self.size_var = tk.StringVar(value="20x20")
        size_menu = ttk.Combobox(size_frame, textvariable=self.size_var, values=["10x10", "15x15", "20x20"], state="readonly", width=10)
        size_menu.pack(side='left', padx=5)
        size_menu.bind("<<ComboboxSelected>>", self.update_board_size)

        # Algorithm selection
        algo_frame = tk.Frame(self.control_frame, bg="#f0f0f0")
        algo_frame.grid(row=1, column=0, columnspan=3, pady=5)
        tk.Label(algo_frame, text="Algorithm:", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(side='left', padx=5)
        self.algorithm_var = tk.StringVar(value="DQN")
        tk.Radiobutton(algo_frame, text="DQN", variable=self.algorithm_var, value="DQN", font=("Arial", 10), bg="#f0f0f0").pack(side='left', padx=5)
        tk.Radiobutton(algo_frame, text="Q-Learning", variable=self.algorithm_var, value="Q-Learning", font=("Arial", 10), bg="#f0f0f0").pack(side='left', padx=5)

        # Mode selection
        mode_frame = tk.Frame(self.control_frame, bg="#f0f0f0")
        mode_frame.grid(row=2, column=0, columnspan=3, pady=5)
        tk.Label(mode_frame, text="Mode:", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(side='left', padx=5)
        self.mode_var = tk.StringVar(value="Human vs Human")
        tk.Radiobutton(mode_frame, text="Human vs Human", variable=self.mode_var, value="Human vs Human", font=("Arial", 10), bg="#f0f0f0").pack(side='left', padx=5)
        tk.Radiobutton(mode_frame, text="Human vs AI", variable=self.mode_var, value="Human vs AI", font=("Arial", 10), bg="#f0f0f0").pack(side='left', padx=5)
        tk.Radiobutton(mode_frame, text="AI vs AI", variable=self.mode_var, value="AI vs AI", font=("Arial", 10), bg="#f0f0f0").pack(side='left', padx=5)

        # Control buttons
        button_style = {"font": ("Arial", 10), "width": 14, "height": 1, "bg": "#0288D1", "fg": "white", "activebackground": "#0277BD"}
        tk.Button(self.control_frame, text="Train AI", command=self.train_ai, **button_style).grid(row=3, column=0, pady=5, padx=5)
        tk.Button(self.control_frame, text="Stop Training", command=self.stop_training, **button_style).grid(row=3, column=1, pady=5, padx=5)
        tk.Button(self.control_frame, text="Save AI", command=self.save_ai, **button_style).grid(row=3, column=2, pady=5, padx=5)
        tk.Button(self.control_frame, text="Load AI", command=self.load_ai, **button_style).grid(row=4, column=0, pady=5, padx=5)
        tk.Button(self.control_frame, text="Plot Results", command=self.plot_results, **button_style).grid(row=4, column=1, pady=5, padx=5)
        tk.Button(self.control_frame, text="Restart", command=self.restart, **button_style).grid(row=4, column=2, pady=5, padx=5)
        tk.Button(self.control_frame, text="Undo Move", command=self.undo_move, **button_style).grid(row=5, column=0, pady=5, padx=5)
        tk.Button(self.control_frame, text="AI vs AI", command=self.ai_vs_ai, **button_style).grid(row=5, column=1, pady=5, padx=5)
        tk.Button(self.control_frame, text="Exit", command=self.exit_game, **button_style).grid(row=5, column=2, pady=5, padx=5)

        # Progress bar
        self.progress_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.progress_frame.pack(pady=5)
        self.progress_label = tk.Label(self.progress_frame, text="Training Progress: 0%", font=("Arial", 12), bg="#f0f0f0")
        self.progress_label.pack(side='left', padx=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(side='left', padx=5)

        # Status and turn label
        self.status_label = tk.Label(self.root, text="Ready", font=("Arial", 12), bd=1, relief='sunken', anchor='w', bg="#f0f0f0")
        self.status_label.pack(side='bottom', fill='x', pady=5)
        self.turn_label = tk.Label(self.root, text="Lượt của X", font=("Arial", 14, "bold"), bg="#f0f0f0", fg="#333")
        self.turn_label.pack(pady=5)

        # Board frame with scrollbars
        self.board_container = tk.Frame(self.root, bg="#f0f0f0")
        self.board_container.pack(fill='both', expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.board_container, width=650, height=450, bg="#ffffff", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(side='left', fill='both', expand=True)

        self.scroll_x = tk.Scrollbar(self.board_container, orient='horizontal', command=self.canvas.xview)
        self.scroll_x.pack(side='bottom', fill='x')
        self.scroll_y = tk.Scrollbar(self.board_container, orient='vertical', command=self.canvas.yview)
        self.scroll_y.pack(side='right', fill='y')

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        self.board_frame = tk.Frame(self.canvas, bg="#ffffff")
        self.canvas.create_window((0, 0), window=self.board_frame, anchor='nw')

        self.update_board_ui()
        logging.info("Board UI setup completed")
        print("Board UI setup completed")

    def update_board_ui(self):
        # Clear existing buttons
        for row in self.buttons:
            for button in row:
                button.destroy()
        self.buttons = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.board_frame.destroy()
        self.board_frame = tk.Frame(self.canvas, bg="#ffffff")
        self.canvas.create_window((0, 0), window=self.board_frame, anchor='nw')

        # Create new board buttons
        for i in range(self.board_size):
            for j in range(self.board_size):
                button = tk.Button(self.board_frame, text=" ", width=3, height=1, font=("Arial", 10),
                                   relief="solid", bg="#e6e6e6", activebackground="#b3b3b3",
                                   command=lambda i=i, j=j: self.on_button_click(i, j))
                button.grid(row=i+1, column=j+1, padx=1, pady=1)
                self.buttons[i][j] = button
            tk.Label(self.board_frame, text=str(i), font=("Arial", 8), bg="#ffffff").grid(row=i+1, column=0, padx=2)
            tk.Label(self.board_frame, text=str(i), font=("Arial", 8), bg="#ffffff").grid(row=i+1, column=self.board_size+1, padx=2)
        for j in range(self.board_size):
            tk.Label(self.board_frame, text=str(j), font=("Arial", 8), bg="#ffffff").grid(row=0, column=j+1, padx=2)
            tk.Label(self.board_frame, text=str(j), font=("Arial", 8), bg="#ffffff").grid(row=self.board_size+1, column=j+1, padx=2)

        self.board_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.env.render(self.buttons)

    def update_board_size(self, event=None):
        size_str = self.size_var.get()
        new_size = int(size_str.split('x')[0])
        if new_size != self.board_size:
            self.board_size = new_size
            self.env = TicTacToeEnv(board_size=self.board_size)
            self.dqn_agent = DQNAgent(board_size=self.board_size)
            self.qlearning_agent = QLearningAgent(board_size=self.board_size)
            self.move_history.clear()
            self.current_player = 1
            self.update_board_ui()
            self.update_turn_label()
            self.status_label.config(text=f"Board size changed to {self.board_size}x{self.board_size}")
            logging.info(f"Board size changed to {self.board_size}x{self.board_size}")

    def check_status_queue(self):
        try:
            while True:
                message = self.status_queue.get_nowait()
                if message.startswith("progress:"):
                    _, progress, status = message.split(":", 2)
                    self.progress_bar['value'] = float(progress)
                    self.progress_label.config(text=f"Training Progress: {float(progress):.1f}%")
                    self.status_label.config(text=status)
                else:
                    self.status_label.config(text=message)
                    self.progress_bar['value'] = 0
                    self.progress_label.config(text="Training Progress: 0%")
                logging.info(f"Status updated: {message}")
        except queue.Empty:
            pass
        self.root.after(100, self.check_status_queue)

    def on_button_click(self, i, j):
        logging.info(f"Button clicked at ({i}, {j})")
        print(f"Button clicked at ({i}, {j})")
        if self.env.board[i, j] != 0:
            return
        if self.mode_var.get() == "Human vs Human":
            self.env.step((i, j), self.current_player)
            self.move_history.append((i, j, self.current_player))
            self.env.render(self.buttons)
            if self.env.done:
                self.show_result()
            else:
                self.current_player = -1 if self.current_player == 1 else 1
                self.update_turn_label()
        elif self.mode_var.get() == "Human vs AI" and self.current_player == -1:
            self.env.step((i, j), -1)
            self.move_history.append((i, j, -1))
            self.env.render(self.buttons)
            if self.env.done:
                self.show_result()
            else:
                self.current_player = 1
                self.agent_move()

    def agent_move(self):
        logging.info("Agent move triggered")
        print("Agent move triggered")
        valid_actions = self.env.get_valid_actions()
        if self.algorithm_var.get() == "DQN":
            action = self.dqn_agent.choose_action(self.env.board, valid_actions)
        else:
            action = self.qlearning_agent.play(self.env.board)
        self.env.step(action, 1)
        self.move_history.append((action[0], action[1], 1))
        self.env.render(self.buttons)
        if self.env.done:
            self.show_result()
        else:
            self.current_player = -1
            self.update_turn_label()

    def ai_vs_ai(self):
        logging.info("AI vs AI triggered")
        print("AI vs AI triggered")
        if self.is_training:
            self.status_label.config(text="Đang huấn luyện, vui lòng đợi!")
            return
        self.status_label.config(text="Đang chạy AI vs AI...")
        self.training_thread = threading.Thread(target=self.ai_vs_ai_thread)
        self.training_thread.daemon = True
        self.training_thread.start()

    def ai_vs_ai_thread(self):
        try:
            env = TicTacToeEnv(board_size=self.board_size)
            results = []
            num_games = 100  # Number of games to play
            for game in range(num_games):
                state = env.reset()
                done = False
                current_player = 1  # DQN starts as X
                moves = []
                while not done:
                    valid_actions = env.get_valid_actions()
                    if current_player == 1:
                        action = self.dqn_agent.choose_action(state, valid_actions)
                    else:
                        action = self.qlearning_agent.choose_action(state, valid_actions)
                    moves.append((action, current_player))
                    state, reward, done = env.step(action, current_player)
                    current_player = -1 if current_player == 1 else 1
                winner = env.winner
                result = "Draw" if winner is None else ("DQN" if winner == 1 else "Q-Learning")
                results.append({"game": game + 1, "winner": result, "moves": len(moves)})
                if (game + 1) % 10 == 0:
                    self.status_queue.put(f"progress:{(game + 1) / num_games * 100:.1f}:AI vs AI Game {game + 1}/{num_games}")

            # Save results to CSV
            with open('aivsai_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["game", "winner", "moves"])
                writer.writeheader()
                writer.writerows(results)
            self.status_queue.put("AI vs AI completed. Results saved to aivsai_results.csv")
        except Exception as e:
            self.status_queue.put(f"Lỗi AI vs AI: {e}")
        finally:
            self.is_training = False

    def update_turn_label(self):
        player = "X" if self.current_player == 1 else "O"
        self.turn_label.config(text=f"Lượt của {player}")
        logging.info(f"Turn updated: {player}")

    def show_result(self):
        logging.info("Showing result")
        print("Showing result")
        winner = self.env.winner
        if winner is None:
            messagebox.showinfo("Kết quả", "Trò chơi hòa!")
        else:
            player = "X" if winner == 1 else "O"
            messagebox.showinfo("Kết quả", f"{player} thắng!")
        self.move_history.clear()
        self.ask_restart()

    def ask_restart(self):
        logging.info("Asking to restart")
        print("Asking to restart")
        result = messagebox.askquestion("Chơi tiếp", "Bạn muốn chơi tiếp không?")
        if result == 'yes':
            self.restart()
        else:
            self.exit_game()

    def restart(self):
        logging.info("Restarting game")
        print("Restarting game")
        self.env.reset()
        self.current_player = 1
        self.move_history.clear()
        self.env.render(self.buttons)
        self.update_turn_label()
        if self.mode_var.get() == "Human vs AI" and self.current_player == 1:
            self.agent_move()

    def undo_move(self):
        logging.info("Undo move triggered")
        print("Undo move triggered")
        if not self.move_history:
            self.status_label.config(text="Không có nước đi để hoàn tác!")
            return
        last_move = self.move_history.pop()
        i, j, player = last_move
        self.env.board[i, j] = 0
        self.env.done = False
        self.env.winner = None
        self.current_player = player
        self.env.render(self.buttons)
        self.update_turn_label()
        self.status_label.config(text="Đã hoàn tác nước đi")

    def train_ai(self):
        if self.is_training:
            self.status_label.config(text="Đang huấn luyện, vui lòng đợi!")
            return
        logging.info("Starting AI training")
        print("Starting AI training")
        self.is_training = True
        self.status_label.config(text="Đang huấn luyện AI...")
        self.training_thread = threading.Thread(target=self.training_thread)
        self.training_thread.daemon = True
        self.training_thread.start()

    def training_thread(self):
        try:
            if self.algorithm_var.get() == "DQN":
                self.dqn_agent.train(episodes=2000, status_queue=self.status_queue)
            else:
                self.qlearning_agent.train(episodes=5000, status_queue=self.status_queue)
            self.status_queue.put("Huấn luyện AI hoàn tất")
        except Exception as e:
            self.status_queue.put(f"Lỗi huấn luyện AI: {e}")
        finally:
            self.is_training = False

    def stop_training(self):
        logging.info("Stop training triggered")
        print("Stop training triggered")
        if self.is_training and self.training_thread and self.training_thread.is_alive():
            self.status_queue.put("Đã yêu cầu hủy huấn luyện AI")
            self.is_training = False
        else:
            self.status_label.config(text="Không có huấn luyện đang chạy")

    def save_ai(self):
        logging.info("Saving AI")
        print("Saving AI")
        try:
            if self.algorithm_var.get() == "DQN":
                self.dqn_agent.save_model(f"dqn_model_{self.board_size}x{self.board_size}.h5")
            else:
                self.qlearning_agent.save_q_table(f"qlearning_qtable_{self.board_size}x{self.board_size}.pkl")
            self.status_label.config(text="Đã lưu AI thành công!")
        except Exception as e:
            self.status_label.config(text=f"Lỗi khi lưu AI: {e}")

    def load_ai(self):
        logging.info("Loading AI")
        print("Loading AI")
        try:
            if self.algorithm_var.get() == "DQN":
                self.dqn_agent.load_model(f"dqn_model_{self.board_size}x{self.board_size}.h5")
            else:
                self.qlearning_agent.load_q_table(f"qlearning_qtable_{self.board_size}x{self.board_size}.pkl")
            self.status_label.config(text="Đã tải AI thành công!")
        except Exception as e:
            self.status_label.config(text=f"Lỗi khi tải AI: {e}")

    def plot_results(self):
        logging.info("Plotting results")
        print("Plotting results")
        try:
            if self.algorithm_var.get() == "DQN":
                self.dqn_agent.plot_results()
                self.status_label.config(text=f"Đã lưu biểu đồ DQN tại dqn_tictactoe_{self.board_size}x{self.board_size}_results.png")
            else:
                self.qlearning_agent.plot_results()
                self.status_label.config(text=f"Đã lưu biểu đồ Q-Learning tại qlearning_tictactoe_{self.board_size}x{self.board_size}_results.png")
        except Exception as e:
            self.status_label.config(text=f"Lỗi khi vẽ biểu đồ: {e}")

    def exit_game(self):
        logging.info("Exiting game")
        print("Exiting game")
        self.root.quit()

if __name__ == "__main__":
    logging.info("Starting main_ui.py")
    print("Starting main_ui.py")
    root = tk.Tk()
    try:
        app = TicTacToeUI(root)
        logging.info("Entering Tkinter mainloop")
        print("Entering Tkinter mainloop")
        root.mainloop()
    except Exception as e:
        logging.error(f"Error in main_ui: {e}")
        print(f"Error in main_ui: {e}")
        raise