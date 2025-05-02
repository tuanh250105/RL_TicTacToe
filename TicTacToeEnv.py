
import numpy as np

class TicTacToeEnv:
    def __init__(self, board_size=20):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.done = False
        self.winner = None
        return self.board

    def is_winner(self, player):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == player:
                    if j + 4 < self.board_size and np.all(self.board[i, j:j+5] == player):
                        return True
                    if i + 4 < self.board_size and np.all(self.board[i:i+5, j] == player):
                        return True
                    if i + 4 < self.board_size and j + 4 < self.board_size and np.all([self.board[i+k, j+k] == player for k in range(5)]):
                        return True
                    if i + 4 < self.board_size and j - 4 >= 0 and np.all([self.board[i+k, j-k] == player for k in range(5)]):
                        return True
        return False

    def get_valid_actions(self):
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i, j] == 0]

    def step(self, action, player):
        if self.done:
            return self.board, 0, True
        i, j = action
        if self.board[i, j] != 0:
            raise ValueError("Ô đã được chiếm đóng, chọn ô khác.")
        self.board[i, j] = player
        if self.is_winner(player):
            self.done = True
            self.winner = player
            return self.board, 1, True
        if not self.get_valid_actions():
            self.done = True
            return self.board, 0, True
        return self.board, 0, False

    def render(self, buttons):
        """Cập nhật giao diện bàn cờ trên các nút Tkinter."""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    buttons[i][j].config(text="X", fg="red")
                elif self.board[i, j] == -1:
                    buttons[i][j].config(text="O", fg="blue")
                else:
                    buttons[i][j].config(text=" ", fg="black")