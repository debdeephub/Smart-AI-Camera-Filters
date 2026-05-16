import cv2
import ollama
import threading
import time

# --- CONFIGURATION ---
MODEL_NAME = 'gemma4:e4b'  # Target exact model: gemma4:e4b
FRAME_WIDTH = 640          # Lower resolution improves processing speed
FRAME_HEIGHT = 480
# ---------------------

class GemmaCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        self.current_frame = None
        self.ai_caption = f"Initializing {MODEL_NAME}..."
        self.running = True
        
        # Start the background thread for Gemma processing
        self.ai_thread = threading.Thread(target=self.async_ai_processor, daemon=True)
        self.ai_thread.start()

    def async_ai_processor(self):
        """Runs in the background to handle heavy vision model processing."""
        while self.running:
            if self.current_frame is not None:
                try:
                    # Save a temporary copy of the latest frame for the VLM
                    temp_filename = 'laptop_frame.jpg'
                    cv2.imwrite(temp_filename, self.current_frame)

                    # Prompt tailored for a quick, creative laptop camera overlay
                    response = ollama.generate(
                        model=MODEL_NAME,
                        prompt="Analyze this camera feed frame. Give a one-sentence, witty overlay caption describing the scene or user vibe.",
                        images=[temp_filename]
                    )
                    
                    self.ai_caption = response['response'].strip()
                except Exception as e:
                    self.ai_caption = f"AI Error: {str(e)}"
            
            # Throttle background requests slightly to prevent your laptop fan from going crazy
            time.sleep(1.5)

    def run(self):
        print(f"Launching Camera with {MODEL_NAME}. Press 'q' to exit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break

            # Mirror the frame for a more natural "selfie" laptop experience
            frame = cv2.flip(frame, 1)
            
            # Update the latest frame for the AI thread to look at
            self.current_frame = frame.copy()

            # --- RENDER OVERLAY LAYER ---
            # Create a semi-transparent black banner at the bottom for readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, FRAME_HEIGHT - 60), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Draw the AI filter text onto the frame
            cv2.putText(
                frame, 
                self.ai_caption, 
                (15, FRAME_HEIGHT - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5,                  # Font scale
                (255, 255, 255),      # White text
                1,                    # Thickness
                cv2.LINE_AA
            )

            # Display the resulting frame
            cv2.imshow(f'{MODEL_NAME} AI Cam', frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = GemmaCamera()
    cam.run()