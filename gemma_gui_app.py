import cv2
import json
import ollama
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk

# Configuration
MODEL_NAME = 'gemma4:e4b'

class GemmaFilterApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Gemma 4 Real-Time AI Camera Engine")
        self.window.geometry("900x550")
        self.window.configure(bg="#1e1e2e")

        # Active filter variables
        self.current_style_label = "None"
        self.filter_config = {
            "cartoon_effect": False, "contrast": 1.0, "brightness": 0, "saturation": 1.0, "color_tint": [1.0, 1.0, 1.0]
        }

        # Initialize Webcam Hardware (Using Mac-safe AVFoundation)
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        # Build UI layout structures
        self.setup_ui()

        # Start Video Capture Loop
        self.running = True
        self.update_video_stream()

    def setup_ui(self):
        # Left Side: Camera Panel Viewfinder
        self.video_frame = tk.Frame(self.window, width=540, height=420, bg="#11111b")
        self.video_frame.pack(side=tk.LEFT, padx=20, pady=20)
        
        self.cam_label = tk.Label(self.video_frame, bg="#11111b")
        self.cam_label.pack()

        # Right Side: Control Panels
        self.control_panel = tk.Frame(self.window, bg="#1e1e2e")
        self.control_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=20)

        # Style Banner Indicator
        self.status_lbl = tk.Label(self.control_panel, text="Active Filter: Original", font=("Helvetica", 12, "bold"), fg="#a6e3a1", bg="#1e1e2e")
        self.status_lbl.pack(anchor="w", pady=5)

        # 1. Preset Filtering Options
        preset_lbl = tk.Label(self.control_panel, text="Preset Studio Filters:", font=("Helvetica", 10, "bold"), fg="#cdd6f4", bg="#1e1e2e")
        preset_lbl.pack(anchor="w", pady=(10, 5))

        presets = [
            ("Japanese Anime", "japanese anime, studio ghibli cel-shaded colorful style"),
            ("Cyberpunk Neon", "cyberpunk neon night, highly saturated magenta and cyan tones, dark contrast"),
            ("90s VHS Tape", "90s vhs camcorder recording, faded warm exposure, vintage retro film tracking style"),
            ("Gothic Noir", "classic noir detective movie, dramatic stark high contrast true black and white"),
            ("Reset View", "original default parameters, clear baseline image, no custom tints")
        ]

        for label, prompt in presets:
            btn = tk.Button(
                self.control_panel, text=label, command=lambda p=prompt, l=label: self.trigger_gemma_thread(p, l),
                bg="#313244", fg="#cdd6f4", activebackground="#45475a", activeforeground="#ffffff",
                relief="flat", pady=6, font=("Helvetica", 9)
            )
            btn.pack(fill=tk.X, pady=3, padx=5)

        # 2. Custom Input Text prompt space
        custom_lbl = tk.Label(self.control_panel, text="Generate Custom Prompt Filter:", font=("Helvetica", 10, "bold"), fg="#cdd6f4", bg="#1e1e2e")
        custom_lbl.pack(anchor="w", pady=(25, 5))

        self.prompt_entry = tk.Entry(self.control_panel, font=("Helvetica", 10), bg="#313244", fg="#cdd6f4", insertbackground="white", borderwidth=0)
        self.prompt_entry.pack(fill=tk.X, pady=5, ipady=6, padx=5)
        self.prompt_entry.insert(0, "")

        self.apply_btn = tk.Button(
            self.control_panel, text="✨ Apply Custom Design with Gemma 4", command=self.apply_custom_prompt,
            bg="#b4befe", fg="#11111b", font=("Helvetica", 9, "bold"), activebackground="#cba6f7", relief="flat", pady=8
        )
        self.apply_btn.pack(fill=tk.X, pady=8, padx=5)

    def apply_custom_prompt(self):
        raw_prompt = self.prompt_entry.get().strip()
        if raw_prompt:
            self.trigger_gemma_thread(raw_prompt, f"Custom: {raw_prompt[:15]}...")

    def trigger_gemma_thread(self, prompt, style_name):
        if style_name == "Reset View":
            self.status_lbl.config(text="Active Filter: Original")
            self.filter_config = {"cartoon_effect": False, "contrast": 1.0, "brightness": 0, "saturation": 1.0, "color_tint": [1.0, 1.0, 1.0]}
            return
            
        self.status_lbl.config(text="🧠 Gemma 4 is calculating settings...")
        threading.Thread(target=self.fetch_filter_logic, args=(prompt, style_name), daemon=True).start()

    def fetch_filter_logic(self, prompt, style_name):
        system_instructions = (
            "You are an automated real-time video filter converter engine. Your job is to translate a user's creative "
            "style request into raw numerical variables for OpenCV processing. You must return ONLY a raw "
            "JSON object with no markdown syntax wrapping blocks, no formatting text, and no conversational noise."
            "\n\nJSON Schema:\n"
            "{\n"
            "  \"cartoon_effect\": true or false,\n"
            "  \"contrast\": float (0.5 to 2.2),\n"
            "  \"brightness\": integer (-40 to 40),\n"
            "  \"saturation\": float (0.0 to 2.5),\n"
            "  \"color_tint\": [B, G, R] float multipliers (0.6 to 1.8)\n"
            "}"
        )

        try:
            response = ollama.generate(
                model=MODEL_NAME, system=system_instructions,
                prompt=f"Create a specific video parameter filter mapping array for: {prompt}",
                options={"temperature": 0.1}
            )
            
            clean_text = response['response'].strip()
            
            # SAFE EXTRACTION: Slice the string directly from the first '{' to the last '}'
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                clean_text = clean_text[start_idx:end_idx]

            self.filter_config = json.loads(clean_text)
            self.window.after(0, lambda: self.status_lbl.config(text=f"Active Filter: {style_name}"))
        except Exception as e:
            self.window.after(0, lambda: self.status_lbl.config(text="❌ Gemma Configuration Failed"))

    def process_pixels(self, frame, cfg):
        frame = cv2.convertScaleAbs(frame, alpha=cfg.get("contrast", 1.0), beta=cfg.get("brightness", 0))

        if cfg.get("saturation", 1.0) != 1.0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype("float32")
            hsv[:, :, 1] *= cfg.get("saturation", 1.0)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            frame = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

        tint = cfg.get("color_tint", [1.0, 1.0, 1.0])
        if tint != [1.0, 1.0, 1.0]:
            frame = frame.astype("float32")
            frame[:, :, 0] *= tint[0] 
            frame[:, :, 1] *= tint[1] 
            frame[:, :, 2] *= tint[2] 
            frame = np.clip(frame, 0, 255).astype("uint8")

        if cfg.get("cartoon_effect", False):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color_smooth = cv2.bilateralFilter(frame, 7, 250, 250)
            frame = cv2.bitwise_and(color_smooth, color_smooth, mask=edges)

        return frame

    def update_video_stream(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (540, 400))
            
            # FIXED LINE: Added 'frame' as the first argument
            processed = self.process_pixels(frame, self.filter_config)

            rgb_img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            tk_img = ImageTk.PhotoImage(image=pil_img)

            self.cam_label.imgtk = tk_img
            self.cam_label.configure(image=tk_img)

        if self.running:
            self.window.after(15, self.update_video_stream)

    def close_app(self):
        self.running = False
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GemmaFilterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()