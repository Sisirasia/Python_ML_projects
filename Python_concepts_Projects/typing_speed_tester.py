import tkinter as tk
from tkinter import messagebox
import random
import time

# Sample text pool
TEXT_POOL = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "Do unto others as you would have them do unto you."
]

class TypingSpeedTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Speed Tester")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        # Variables
        self.text_to_type = random.choice(TEXT_POOL)
        self.start_time = None
        self.end_time = None
        
        # UI Components
        self.setup_ui()

    def setup_ui(self):
        # Title Label
        title_label = tk.Label(self.root, text="Typing Speed Tester", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        # Text to type
        self.text_label = tk.Label(self.root, text=self.text_to_type, font=("Arial", 14), wraplength=500, justify="center")
        self.text_label.pack(pady=10)

        # Typing Area
        self.typing_area = tk.Text(self.root, height=6, width=60, font=("Arial", 12))
        self.typing_area.pack(pady=10)
        self.typing_area.bind("<FocusIn>", self.start_timer)

        # Buttons
        submit_button = tk.Button(self.root, text="Submit", font=("Arial", 12), command=self.calculate_speed)
        submit_button.pack(pady=5)

        reset_button = tk.Button(self.root, text="Reset", font=("Arial", 12), command=self.reset_test)
        reset_button.pack(pady=5)

    def start_timer(self, event=None):
        """Start the timer when the user starts typing."""
        if self.start_time is None:
            self.start_time = time.time()

    def calculate_speed(self):
        """Calculate typing speed and display results."""
        if self.start_time is None:
            messagebox.showwarning("Error", "Start typing to begin the timer!")
            return

        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        typed_text = self.typing_area.get("1.0", tk.END).strip()

        # Calculate metrics
        words_per_minute = len(typed_text.split()) / (elapsed_time / 60)
        accuracy = self.calculate_accuracy(typed_text)

        # Show results
        messagebox.showinfo(
            "Results",
            f"Typing Speed: {words_per_minute:.2f} WPM\n"
            f"Accuracy: {accuracy:.2f}%\n"
            f"Time Taken: {elapsed_time:.2f} seconds"
        )


    def calculate_accuracy(self, typed_text):
        """Calculate typing accuracy."""
        original_words = self.text_to_type.split()
        typed_words = typed_text.split()
        matches = sum(1 for ow, tw in zip(original_words, typed_words) if ow == tw)
        return (matches / len(original_words)) * 100

    def reset_test(self):
        """Reset the test for a new typing session."""
        self.text_to_type = random.choice(TEXT_POOL)
        self.start_time = None
        self.end_time = None
        self.text_label.config(text=self.text_to_type)
        self.typing_area.delete("1.0", tk.END)


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = TypingSpeedTester(root)
    root.mainloop()
