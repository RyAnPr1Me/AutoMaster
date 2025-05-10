import tkinter as tk
from tkinter import filedialog, messagebox
from pydub import AudioSegment
from Master import load_audio, apply_mastering_chain
import os

def browse_input():
    path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if path:
        input_path.set(path)

def browse_output():
    path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV Files", "*.wav")])
    if path:
        output_path.set(path)

def run_mastering():
    in_path = input_path.get()
    out_path = output_path.get()
    if not os.path.exists(in_path):
        messagebox.showerror("Error", "Invalid input file path.")
        return

    try:
        audio = load_audio(in_path)
        mastered = apply_mastering_chain(audio)
        mastered.export(out_path, format="wav")
        messagebox.showinfo("Done", f"Exported mastered file to:\n{out_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Mastering failed:\n{e}")

# === GUI Setup ===
root = tk.Tk()
root.title("AI Mastering Tool")

input_path = tk.StringVar()
output_path = tk.StringVar()

tk.Label(root, text="Input File:").grid(row=0, column=0, padx=10, pady=5, sticky='e')
tk.Entry(root, textvariable=input_path, width=50).grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=browse_input).grid(row=0, column=2, padx=5)

tk.Label(root, text="Output File:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
tk.Entry(root, textvariable=output_path, width=50).grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=browse_output).grid(row=1, column=2, padx=5)

tk.Button(root, text="Run Mastering", command=run_mastering, bg="green", fg="white").grid(row=2, column=1, pady=20)

root.mainloop()
