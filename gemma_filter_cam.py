import cv2
import json
import ollama
import numpy as np

# Configuration
MODEL_NAME = 'gemma4:e4b'

def get_filter_config_from_gemma(user_prompt):
    """
    Uses Gemma 4 to convert a creative text prompt into a structured JSON 
    configuration that OpenCV can use to manipulate image pixels.
    """
    print(f"\n🧠 Gemma 4 is designing your '{user_prompt}' filter pipeline...")
    
    system_instructions = (
        "You are an expert real-time video filter engine. Your job is to translate a user's creative "
        "style request into raw numerical parameters for OpenCV processing. You must return ONLY a raw "
        "JSON object with no markdown blocks, no explanations, and no extra text."
        "\n\nFormat requirements:\n"
        "{\n"
        "  \"cartoon_effect\": true or false (true if style needs outlines/anime look),\n"
        "  \"contrast\": float between 0.5 and 2.5 (1.0 is default),\n"
        "  \"brightness\": integer between -50 and 50 (0 is default),\n"
        "  \"saturation\": float between 0.0 and 3.0 (1.0 is default),\n"
        "  \"color_tint\": [B, G, R] multipliers, floats between 0.5 and 2.0 (e.g., [1.0, 1.0, 1.2] adds warmth)\n"
        "}"
    )

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            system=system_instructions,
            prompt=f"Create a filter configuration for this style: {user_prompt}",
            options={"temperature": 0.2} # Low temperature for reliable structured output
        )
        
        # Parse the JSON response clean of any accidental LLM fluff
        clean_response = response['response'].strip()
        if "```json" in clean_response:
            clean_response = clean_response.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_response:
            clean_response = clean_response.split("```")[1].strip()

        config = json.loads(clean_response)
        print("✅ Filter Configuration Generated Successfully:")
        print(json.dumps(config, indent=2))
        return config

    except Exception as e:
        print(f"❌ Failed to parse Gemma output. Falling back to default settings. Error: {e}")
        return {
            "cartoon_effect": False,
            "contrast": 1.0,
            "brightness": 0,
            "saturation": 1.0,
            "color_tint": [1.0, 1.0, 1.0]
        }

def apply_opencv_filter(frame, cfg):
    """
    Applies mathematical transformations to a video frame using 
    the parameters configured by Gemma 4.
    """
    # 1. Adjust Brightness and Contrast
    frame = cv2.convertScaleAbs(frame, alpha=cfg.get("contrast", 1.0), beta=cfg.get("brightness", 0))

    # 2. Adjust Saturation
    if cfg.get("saturation", 1.0) != 1.0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[:, :, 1] *= cfg.get("saturation", 1.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        frame = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    # 3. Apply Color Tint Matrix
    tint = cfg.get("color_tint", [1.0, 1.0, 1.0])
    if tint != [1.0, 1.0, 1.0]:
        frame = frame.astype("float32")
        frame[:, :, 0] *= tint[0] # Blue channel
        frame[:, :, 1] *= tint[1] # Green channel
        frame[:, :, 2] *= tint[2] # Red channel
        frame = np.clip(frame, 0, 255).astype("uint8")

    # 4. Apply Cell-Shaded Cartoon / Anime Effect (Edging + Color Quantization)
    if cfg.get("cartoon_effect", False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        # Generate thick anime-style line art outlines
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        
        # Smooth colors out to simulate cel-shading paint
        color_smooth = cv2.bilateralFilter(frame, 9, 300, 300)
        
        # Merge outline layer over smoothed colors
        frame = cv2.bitwise_and(color_smooth, color_smooth, mask=edges)

    return frame

def main():
    print("=" * 60)
    print("   GEMMA 4: PROMPT-DRIVEN VIDEO FILTER SYSTEM")
    print("=" * 60)
    
    user_prompt = input("Enter your desired video filter style (e.g., 'japanese anime', 'cyberpunk neon', '90s vintage film'): ")
    if not user_prompt.strip():
        user_prompt = "japanese anime"

    # Fetch configuration parameters dynamically from Gemma
    filter_config = get_filter_config_from_gemma(user_prompt)

    # Boot webcam hardware
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("\n🎥 Stream active. Press 'q' inside the video window to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror video window for natural human perception
        frame = cv2.flip(frame, 1)

        # Process the raw frame utilizing Gemma's generated matrix
        stylized_frame = apply_opencv_filter(frame, filter_config)

        # Draw a tiny text stamp of your prompt onto the layout
        cv2.putText(stylized_frame, f"Filter Active: {user_prompt}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Render window frame
        cv2.imshow("Gemma AI Filter Stream", stylized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()