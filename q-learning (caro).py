import turtle
import random
import time
from tkinter import Tk, Button, Label, messagebox, Checkbutton, IntVar

# ======== THIẾT LẬP MẶC ĐỊNH ========
KICH_THUOC_BAN_CO = 20
SO_QUAN_THANG = 5
TIME_LIMIT = 10  # 10 giây

# ======== CÁC HÀM XỬ LÝ CỜ ========
def tao_ban_co_trong(sz=KICH_THUOC_BAN_CO):
    return [[" "] * sz for _ in range(sz)]

def trong_ban_co(board, y, x):
    return 0 <= y < len(board) and 0 <= x < len(board)

def dem_quan(board, y, x, dy, dx, quan):
    dem = 0
    for _ in range(5):
        if trong_ban_co(board, y, x) and board[y][x] == quan:
            dem += 1
            y += dy
            x += dx
        else:
            break
    return dem

def kiem_tra_thang(board):
    size = len(board)
    for y in range(size):
        for x in range(size):
            if board[y][x] != ' ':
                for dy, dx in [(0,1), (1,0), (1,1), (1,-1)]:
                    if dem_quan(board, y, x, dy, dx, board[y][x]) >= SO_QUAN_THANG:
                        return f"{board[y][x]} thắng"
    if all(board[i][j] != ' ' for i in range(size) for j in range(size)):
        return "Hòa"
    return "Chơi tiếp"

def ai_danh(board):
    def tim_nuoc(quan, so_quan, can_2_dau=False):
        for y in range(KICH_THUOC_BAN_CO):
            for x in range(KICH_THUOC_BAN_CO):
                if board[y][x] != quan:
                    continue
                for dy, dx in [(0,1), (1,0), (1,1), (1,-1)]:
                    count = 1
                    ny, nx = y + dy, x + dx
                    while trong_ban_co(board, ny, nx) and board[ny][nx] == quan:
                        count += 1
                        ny += dy
                        nx += dx
                    if count == so_quan:
                        truoc_y, truoc_x = y - dy, x - dx
                        sau_y, sau_x = ny, nx
                        if can_2_dau:
                            if trong_ban_co(board, truoc_y, truoc_x) and board[truoc_y][truoc_x] == ' ' and trong_ban_co(board, sau_y, sau_x) and board[sau_y][sau_x] == ' ':
                                return (truoc_y, truoc_x)
                        else:
                            if trong_ban_co(board, truoc_y, truoc_x) and board[truoc_y][truoc_x] == ' ':
                                return (truoc_y, truoc_x)
                            if trong_ban_co(board, sau_y, sau_x) and board[sau_y][sau_x] == ' ':
                                return (sau_y, sau_x)
        return None

    # 1. Thắng ngay nếu được
    nuoc = tim_nuoc('O', 4)
    if nuoc:
        return nuoc

    # 2. Chặn đối thủ nếu họ sắp thắng (4 quân cần chặn 2 đầu)
    nuoc = tim_nuoc('X', 4, can_2_dau=True)
    if nuoc:
        return nuoc

    # 3. Chặn 3 quân 1 đầu
    nuoc = tim_nuoc('X', 3)
    if nuoc:
        return nuoc

    # 4. Chiến lược tấn công: nối dài chuỗi 'O'
    diem_tot = []
    for y in range(KICH_THUOC_BAN_CO):
        for x in range(KICH_THUOC_BAN_CO):
            if board[y][x] == ' ':
                score = 0
                for dy, dx in [(0,1), (1,0), (1,1), (1,-1)]:
                    ny, nx = y + dy, x + dx
                    if trong_ban_co(board, ny, nx) and board[ny][nx] == 'O':
                        score += 2
                    ny2, nx2 = y - dy, x - dx
                    if trong_ban_co(board, ny2, nx2) and board[ny2][nx2] == 'O':
                        score += 2
                    # kiểm tra gần đối thủ để chặn tạm thời
                    if trong_ban_co(board, ny, nx) and board[ny][nx] == 'X':
                        score += 1
                    if trong_ban_co(board, ny2, nx2) and board[ny2][nx2] == 'X':
                        score += 1
                if score > 0:
                    diem_tot.append((score, (y, x)))
    if diem_tot:
        diem_tot.sort(reverse=True)
        return diem_tot[0][1]

    # 5. Nếu không còn nước tốt, chọn random gần các ô đã đánh
    diem_phu = []
    for y in range(KICH_THUOC_BAN_CO):
        for x in range(KICH_THUOC_BAN_CO):
            if board[y][x] == ' ':
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if trong_ban_co(board, ny, nx) and board[ny][nx] != ' ':
                            diem_phu.append((y, x))
                            break
    if diem_phu:
        return random.choice(diem_phu)

    diem_con_lai = [(i, j) for i in range(KICH_THUOC_BAN_CO) for j in range(KICH_THUOC_BAN_CO) if board[i][j] == ' ']
    return random.choice(diem_con_lai) if diem_con_lai else None

# ======== GIAO DIỆN GAME ========
ban_co = tao_ban_co_trong()
history_moves = []
chien_thang = False
luot_nguoi = 'X'
che_do = "nguoi_vs_ai"
highlight_oto = None
timer_id = None

root = Tk()
root.title("Cờ Caro 20x20")

bat_timer = IntVar()

status_label = Label(root, text="Đang chơi...", font=("Arial",12), bd=1, relief="sunken", anchor="w")
status_label.pack(side="bottom", fill="x")

screen = turtle.Screen()
screen.setup(width=800, height=800)
screen.title("Bàn cờ")
screen.tracer(0)
pen = turtle.Turtle()
pen.hideturtle()
pen.speed(0)

size_pixel = 800
o_size = size_pixel // KICH_THUOC_BAN_CO
screen.setworldcoordinates(0, size_pixel, size_pixel, 0)
screen.bgcolor("#ffffff")

def ve_ban_co():
    pen.clear()
    pen.color("black")
    for i in range(KICH_THUOC_BAN_CO + 1):
        pen.penup()
        pen.goto(0, i * o_size)
        pen.pendown()
        pen.goto(size_pixel, i * o_size)
    for i in range(KICH_THUOC_BAN_CO + 1):
        pen.penup()
        pen.goto(i * o_size, 0)
        pen.pendown()
        pen.goto(i * o_size, size_pixel)
    for y in range(KICH_THUOC_BAN_CO):
        for x in range(KICH_THUOC_BAN_CO):
            if ban_co[y][x] != ' ':
                ve_dau(x, y, ban_co[y][x])
    screen.update()

def ve_dau(x, y, dau):
    pen.penup()
    pen.goto(x * o_size + o_size // 2, y * o_size + o_size // 2)
    pen.color("red" if dau == 'X' else "blue")
    pen.write(dau, align="center", font=("Arial", 14, "bold"))
    screen.update()

def ket_thuc(ket_qua):
    global chien_thang
    chien_thang = True
    status_label.config(text=ket_qua)
    messagebox.showinfo("Kết quả", ket_qua)

def undo():
    global history_moves, ban_co, chien_thang
    if not history_moves:
        return
    last = history_moves.pop()
    ban_co[last[0]][last[1]] = ' '
    if che_do == "nguoi_vs_ai" and history_moves:
        last2 = history_moves.pop()
        ban_co[last2[0]][last2[1]] = ' '
    chien_thang = False
    ve_ban_co()

def chuyen_che_do():
    global che_do
    che_do = "nguoi_vs_nguoi" if che_do == "nguoi_vs_ai" else "nguoi_vs_ai"
    khoi_dong()

def restart():
    khoi_dong()

def khoi_dong():
    global ban_co, history_moves, chien_thang, luot_nguoi
    ban_co = tao_ban_co_trong()
    history_moves = []
    chien_thang = False
    luot_nguoi = 'X'
    ve_ban_co()
    status_label.config(text="Đang chơi...")

def het_gio():
    if not chien_thang:
        status_label.config(text="Hết giờ!")

def bat_dong_ho():
    global timer_id
    if bat_timer.get():
        timer_id = root.after(TIME_LIMIT * 1000, het_gio)

def huy_dong_ho():
    global timer_id
    if timer_id:
        root.after_cancel(timer_id)

def click(x, y):
    global chien_thang, luot_nguoi
    if chien_thang:
        return
    cot = int(x // o_size)
    hang = int(y // o_size)
    if trong_ban_co(ban_co, hang, cot) and ban_co[hang][cot] == ' ':
        ban_co[hang][cot] = luot_nguoi
        history_moves.append((hang, cot))
        ve_ban_co()
        ket_qua = kiem_tra_thang(ban_co)
        if ket_qua != "Chơi tiếp":
            ket_thuc(ket_qua)
            return
        if che_do == "nguoi_vs_ai":
            nuoc_ai = ai_danh(ban_co)
            if nuoc_ai:
                ban_co[nuoc_ai[0]][nuoc_ai[1]] = 'O'
                history_moves.append((nuoc_ai[0], nuoc_ai[1]))
                ve_ban_co()
                ket_qua = kiem_tra_thang(ban_co)
                if ket_qua != "Chơi tiếp":
                    ket_thuc(ket_qua)
        else:
            luot_nguoi = 'O' if luot_nguoi == 'X' else 'X'
        huy_dong_ho()
        bat_dong_ho()

ve_ban_co()
screen.onclick(click)

Button(root, text="Undo", command=undo).pack(side="left")
Button(root, text="Chuyển chế độ", command=chuyen_che_do).pack(side="left")
Button(root, text="Chơi lại", command=restart).pack(side="left")
Checkbutton(root, text="Bật đếm giờ", variable=bat_timer).pack(side="left")
Button(root, text="Thoát", command=root.destroy).pack(side="right")

root.mainloop()

