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
        self.window.title("Studio Cam")
        self.window.geometry("1000x650")
        
        # Apple System Dark Gray Palette
        self.bg_color = "#1C1C1E"       # SystemBackground
        self.panel_color = "#2C2C2E"    # SecondarySystemBackground
        self.accent_color = "#0A84FF"   # Apple System Blue
        self.text_color = "#000000"     # Primary Text
        self.text_dim = "#AEAEB2"       # Secondary Text / Labels
        self.success_color = "#30D158"  # Apple System Green

        self.window.configure(bg=self.bg_color)

        # Active filter configurations
        self.filter_config = {
            "cartoon_effect": False, "contrast": 1.0, "brightness": 0, "saturation": 1.0, "color_tint": [1.0, 1.0, 1.0]
        }

        # Initialize Webcam Hardware
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        
        self.setup_ui()
        self.running = True
        self.update_video_stream()

    def setup_ui(self):
        # Top Header Bar
        header_frame = tk.Frame(self.window, bg=self.bg_color)
        header_frame.pack(fill=tk.X, padx=30, pady=(25, 10))
        
        title_lbl = tk.Label(header_frame, text="Studio Lens", font=("SF Pro Display", 22, "bold"), fg=self.text_color, bg=self.bg_color)
        title_lbl.pack(side=tk.LEFT)
        
        # Subtitle or active state indicator (FIXED: "medium" -> "normal")
        self.status_lbl = tk.Label(header_frame, text="Original View", font=("SF Pro Text", 12, "normal"), fg=self.success_color, bg=self.bg_color)
        self.status_lbl.pack(side=tk.RIGHT, pady=(5, 0))

        # Main Split Body Container
        main_container = tk.Frame(self.window, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # LEFT PANEL: The Viewfinder Wrapper (Sleek Apple Matte Frame)
        self.video_frame = tk.Frame(main_container, bg=self.panel_color, bd=0)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15), pady=10)
        
        self.cam_label = tk.Label(self.video_frame, bg=self.panel_color)
        self.cam_label.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

        # RIGHT PANEL: The Floating Sidebar Control Sheet
        sidebar = tk.Frame(main_container, width=320, bg=self.panel_color)
        sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(15, 0), pady=10)
        sidebar.pack_propagate(False)

        # Section 1: Presets Label
        preset_lbl = tk.Label(sidebar, text="PRESETS", font=("SF Pro Text", 10, "bold"), fg=self.text_dim, bg=self.panel_color)
        preset_lbl.pack(anchor="w", padx=20, pady=(20, 10))

        presets = [
            ("Japanese Anime", "japanese anime, studio ghibli cel-shaded colorful style"),
            ("Cyberpunk Neon", "cyberpunk neon night, highly saturated magenta and cyan tones, dark contrast"),
            ("90s VHS Tape", "90s vhs camcorder recording, faded warm exposure, vintage retro film tracking style"),
            ("Gothic Noir", "classic noir detective movie, dramatic stark high contrast true black and white"),
            ("Reset Engine", "original default parameters, clear baseline image, no custom tints")
        ]

        for label, prompt in presets:
            is_reset = (label == "Reset Engine")
            btn_bg = "#3A3A3C" if not is_reset else "#48484A"
            btn_fg = self.text_color if not is_reset else "#FF453A"  
            
            # FIXED: "medium" -> "normal"
            btn = tk.Button(
                sidebar, text=label, command=lambda p=prompt, l=label: self.trigger_gemma_thread(p, l),
                bg=btn_bg, fg=btn_fg, activebackground="#48484A", activeforeground=self.text_color,
                relief="flat", borderwidth=0, font=("SF Pro Text", 11, "normal"), pady=10, cursor="hand2"
            )
            btn.pack(fill=tk.X, pady=4, padx=20)

        # Separator Line
        sep = tk.Frame(sidebar, height=1, bg="#3A3A3C")
        sep.pack(fill=tk.X, padx=20, pady=20)

        # Section 2: Custom Prompting Frame
        custom_lbl = tk.Label(sidebar, text="CREATIVE INPUT", font=("SF Pro Text", 10, "bold"), fg=self.text_dim, bg=self.panel_color)
        custom_lbl.pack(anchor="w", padx=20, pady=(0, 10))

        # Modern Search-bar-like Input entry field
        self.prompt_entry = tk.Entry(
            sidebar, font=("SF Pro Text", 11), bg="#3A3A3C", fg=self.text_color,
            insertbackground="white", borderwidth=0, highlightthickness=1, highlightbackground="#3A3A3C",
            highlightcolor=self.accent_color
        )
        self.prompt_entry.pack(fill=tk.X, pady=5, ipady=10, padx=20)
        
        # Prompt placeholder management
        self.prompt_entry.insert(0, "Describe a filter vibe...")
        self.prompt_entry.bind("<FocusIn>", lambda e: self.prompt_entry.delete(0, tk.END) if self.prompt_entry.get() == "Describe a filter vibe..." else None)

        # High-accent Blue Apply Button
        self.apply_btn = tk.Button(
            sidebar, text="Generate Filter", command=self.apply_custom_prompt,
            bg=self.accent_color, fg=self.text_color, font=("SF Pro Text", 11, "bold"),
            activebackground="#007AFF", activeforeground=self.text_color, relief="flat", borderwidth=0, pady=12, cursor="hand2"
        )
        self.apply_btn.pack(fill=tk.X, pady=12, padx=20)

    def apply_custom_prompt(self):
        raw_prompt = self.prompt_entry.get().strip()
        if raw_prompt and raw_prompt != "Describe a filter vibe...":
            self.trigger_gemma_thread(raw_prompt, f"“{raw_prompt[:12]}...”")

    def trigger_gemma_thread(self, prompt, style_name):
        if style_name == "Reset Engine":
            self.status_lbl.config(text="Original View", fg=self.success_color)
            self.filter_config = {"cartoon_effect": False, "contrast": 1.0, "brightness": 0, "saturation": 1.0, "color_tint": [1.0, 1.0, 1.0]}
            return
            
        self.status_lbl.config(text="Generating with Gemma 4...", fg=self.accent_color)
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
            
            start_idx = clean_text.find('{')
            end_idx = clean_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                clean_text = clean_text[start_idx:end_idx]

            self.filter_config = json.loads(clean_text)
            self.window.after(0, lambda: self.status_lbl.config(text=f"Active: {style_name}", fg=self.success_color))
        except Exception as e:
            self.window.after(0, lambda: self.status_lbl.config(text="Generation Error", fg="#FF453A"))

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
            frame = cv2.resize(frame, (620, 460))
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