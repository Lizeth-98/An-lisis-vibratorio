import time
import pyautogui
import tkinter as tk
import threading
 
def dontsleep():
    
    while True:
        try:
            pyautogui.press('esc') 
            time.sleep(40) 
        except:
            time.sleep(40)
 
def start_dontsleep_thread():
    
    thread = threading.Thread(target=dontsleep, daemon=True)
    thread.start()
 
def KeepUI():
    
    root = tk.Tk()
    root.title("Keep-Me-Up")
    root.geometry("400x150")
    root.resizable(False, False)
 
    
    label = tk.Label(
        root,
        text="Keep-Me-Up se está ejecutando.\n"
             "Puedes mantenerlo minimizado y el programa seguirá ejecutándose.\n"
             "Cierra el programa para detener su función.",
        wraplength=350,
        justify="center"
    )
    label.pack(pady=20)
 
   
    def on_close():
        root.destroy()  
 
    root.protocol("WM_DELETE_WINDOW", on_close)  
 
   
    start_dontsleep_thread()
 
    
    root.mainloop()
 
if __name__ == '__main__':
    KeepUI()