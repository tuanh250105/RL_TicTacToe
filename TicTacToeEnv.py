import tkinter as tk
from tkinter import messagebox
import numpy as np

class TicTacToeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe - 20x20")
        
        # Khởi tạo môi trường Tic-Tac-Toe 20x20
        self.env = TicTacToeEnv(board_size=20)
        
        # Lưu thông tin trạng thái trò chơi
        self.current_player = 1  # Người chơi 1 bắt đầu
        self.buttons = [[None for _ in range(20)] for _ in range(20)]
        
        # Tạo các nút trên giao diện
        self.create_buttons()
        
    def create_buttons(self):
        """Tạo các nút cho bàn cờ 20x20."""
        for i in range(20):
            for j in range(20):
                button = tk.Button(self.root, text=" ", width=4, height=2, font=("Arial", 16),
                                   relief="solid", bg="#e6e6e6", activebackground="#b3b3b3",
                                   command=lambda i=i, j=j: self.on_button_click(i, j))
                button.grid(row=i, column=j, padx=2, pady=2)
                self.buttons[i][j] = button

    def on_button_click(self, i, j):
        """Xử lý khi người chơi nhấn vào một ô."""
        if self.env.board[i, j] != 0:
            return  # Ô đã được đánh dấu, không làm gì cả
        
        # Cập nhật trạng thái bàn cờ
        self.env.step((i, j), self.current_player)
        self.buttons[i][j].config(text="X" if self.current_player == 1 else "O", fg="red" if self.current_player == 1 else "blue")
        
        # Kiểm tra kết quả
        if self.env.done:
            self.show_result()
        else:
            self.switch_player()
        
    def switch_player(self):
        """Chuyển lượt cho người chơi tiếp theo."""
        self.current_player = -1 if self.current_player == 1 else 1
        
    def show_result(self):
        """Hiển thị kết quả của trò chơi (thắng, hòa)."""
        winner = self.env.winner
        if winner is None:
            messagebox.showinfo("Kết quả", "Trò chơi hòa!")
        else:
            player = "X" if winner == 1 else "O"
            messagebox.showinfo("Kết quả", f"Người chơi {player} thắng!")
        
        # Hỏi người chơi có muốn chơi tiếp hay không
        self.ask_restart()

    def ask_restart(self):
        """Hiển thị hộp thoại yêu cầu người chơi chọn tiếp tục hay thoát."""
        result = messagebox.askquestion("Chơi tiếp", "Bạn muốn chơi tiếp không?")
        if result == 'yes':
            self.reset_game()
        else:
            self.root.quit()  # Thoát trò chơi

    def reset_game(self):
        """Đặt lại trò chơi."""
        self.env.reset()
        for i in range(20):
            for j in range(20):
                self.buttons[i][j].config(text=" ", fg="black")
        self.current_player = 1  # Người chơi 1 bắt đầu lại

# Lớp môi trường Tic-Tac-Toe với bàn cờ 20x20
class TicTacToeEnv:
    def __init__(self, board_size=20):
        self.board_size = board_size
        self.reset()

    def reset(self):
        """Khởi tạo bàn cờ và trả về trạng thái ban đầu."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)  # Bàn cờ 20x20
        self.done = False  # Trạng thái trò chơi (chưa kết thúc)
        self.winner = None  # Người thắng (nếu có)
        return self.board

    def is_winner(self, player):
        """Kiểm tra xem người chơi có thắng không (player = 1 hoặc -1)."""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == player:
                    # Kiểm tra ngang
                    if j + 4 < self.board_size and np.all(self.board[i, j:j+5] == player):
                        return True
                    # Kiểm tra dọc
                    if i + 4 < self.board_size and np.all(self.board[i:i+5, j] == player):
                        return True
                    # Kiểm tra chéo phải xuống
                    if i + 4 < self.board_size and j + 4 < self.board_size and np.all([self.board[i+k, j+k] == player for k in range(5)]):
                        return True
                    # Kiểm tra chéo trái xuống
                    if i + 4 < self.board_size and j - 4 >= 0 and np.all([self.board[i+k, j-k] == player for k in range(5)]):
                        return True
        return False

    def get_valid_actions(self):
        """Trả về danh sách các ô trống (hành động hợp lệ)."""
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i, j] == 0]

    def step(self, action, player):
        """Cập nhật trạng thái bàn cờ sau một nước đi."""
        if self.done:
            return self.board, 0, True  # Trò chơi đã kết thúc

        i, j = action
        if self.board[i, j] != 0:
            raise ValueError("Ô đã được chiếm đóng, chọn ô khác.")

        self.board[i, j] = player  # Người chơi đi vào ô

        # Kiểm tra thắng
        if self.is_winner(player):
            self.done = True
            self.winner = player
            return self.board, 1, True  # Trả về thắng cho người chơi

        # Kiểm tra hòa (không còn ô trống)
        if not self.get_valid_actions():
            self.done = True
            return self.board, 0, True  # Hòa, không ai thắng

        return self.board, 0, False  # Trò chơi vẫn tiếp tục

# Khởi động ứng dụng Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeApp(root)
    root.mainloop()
