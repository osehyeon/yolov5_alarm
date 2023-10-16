import tkinter as tk

def main():
    root = tk.Tk()
    root.title("Miracle Morning")

    # 제목 레이블 추가
    title_label = tk.Label(root, text="Miracle Morning", font=("Arial", 24))
    title_label.pack(pady=20)

    # 시계 레이블 추가
    time_label = tk.Label(root, text="15:50:18", font=("Arial", 48))
    time_label.pack(pady=20)

    # 드롭다운 메뉴 및 스핀박스 추가
    dropdown = tk.StringVar(root)
    dropdown.set("오전")  # default value
    dropdown_menu = tk.OptionMenu(root, dropdown, "오전", "오후")
    dropdown_menu.pack(pady=10, side=tk.LEFT, padx=5)

    spinbox_hour = tk.Spinbox(root, from_=0, to=23, width=5)
    spinbox_hour.pack(pady=10, side=tk.LEFT, padx=5)

    spinbox_min = tk.Spinbox(root, from_=0, to=59, width=5)
    spinbox_min.pack(pady=10, side=tk.LEFT, padx=5)

    spinbox_sec = tk.Spinbox(root, from_=0, to=59, width=5)
    spinbox_sec.pack(pady=10, side=tk.LEFT, padx=5)

    # 텍스트 입력 추가
    text_entry = tk.Entry(root, width=50)
    text_entry.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
