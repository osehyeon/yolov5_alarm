import tkinter as tk
from datetime import datetime
import threading
import time

class MiracleMorningUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Miracle Morning")

        # Application Title
        self.title_label = tk.Label(root, text="Miracle Morning", font=("Arial", 20))
        self.title_label.pack(pady=20)

        # Display Current Time
        self.current_time_label = tk.Label(root, text="", font=("Arial", 24))
        self.current_time_label.pack(pady=20)
        self.update_time()

        # Alarm Start Time
        self.start_hour_var = tk.StringVar(root)
        self.start_hour_var.set("오전")
        self.start_hour_dropdown = tk.OptionMenu(root, self.start_hour_var, "오전", "오후")
        self.start_hour_dropdown.pack(side=tk.LEFT, padx=5)

        self.start_time_var = tk.StringVar(root)
        self.start_time_entry = tk.Entry(root, textvariable=self.start_time_var, width=5, font=("Arial", 14))
        self.start_time_entry.pack(side=tk.LEFT, padx=5)
        self.start_time_var.set("07:00")

        # Alarm End Time (if needed in the future)
        self.end_hour_var = tk.StringVar(root)
        self.end_hour_var.set("오전")
        self.end_hour_dropdown = tk.OptionMenu(root, self.end_hour_var, "오전", "오후")
        self.end_hour_dropdown.pack(side=tk.LEFT, padx=5)

        self.end_time_var = tk.StringVar(root)
        self.end_time_entry = tk.Entry(root, textvariable=self.end_time_var, width=5, font=("Arial", 14))
        self.end_time_entry.pack(side=tk.LEFT, padx=5)
        self.end_time_var.set("08:00")

        # Confirmation Button
        self.confirm_button = tk.Button(root, text="확인", command=self.confirm_alarm_time, font=("Arial", 14))
        self.confirm_button.pack(pady=20)

    def update_time(self):
        current_time = datetime.now().strftime('%H:%M:%S')
        self.current_time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def confirm_alarm_time(self):
        start_hour = self.start_time_var.get()
        end_hour = self.end_time_var.get()
        start_am_pm = self.start_hour_var.get()
        end_am_pm = self.end_hour_var.get()

        if start_am_pm == "오후" and start_hour != "12":
            start_hour = str(int(start_hour.split(':')[0]) + 12) + ":" + start_hour.split(':')[1]
        if end_am_pm == "오후" and end_hour != "12":
            end_hour = str(int(end_hour.split(':')[0]) + 12) + ":" + end_hour.split(':')[1]

        print(f"Alarm Start Time: {start_hour}")
        print(f"Alarm End Time: {end_hour}")
        # Add yolov5 inference code here if required

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = MiracleMorningUI(root)
    app.run()