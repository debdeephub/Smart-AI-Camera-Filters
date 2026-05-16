import cv2
import time

print("Scanning for working camera devices...")

found_working_cam = False

for index in range(5):
    print(f"\nTesting Camera Index: {index}...")
    
    # Force the AVFoundation backend natively used by macOS
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print(f"❌ Index {index} could not be opened at all.")
        continue
        
    # Apply a hardware warmup delay required by Apple Silicon
    time.sleep(1.0)
    
    # Try to grab a frame
    ret, frame = cap.read()
    
    if ret and frame is not None:
        print(f"✅ SUCCESS! Index {index} is feeding live video data.")
        print("Launching live test window... Press 'q' to exit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f'Working Camera - Index {index}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        found_working_cam = True
        break
    else:
        print(f"⚠️ Index {index} opened, but returned blank/empty frames.")
    
    cap.release()

if not found_working_cam:
    print("\n❌ All indices checked. No video stream could be read.")
    print("This confirms a hard macOS Privacy/Sandbox lock.")