import cv2

def capture_video():
    # 1. Initialize the webcam (0 is usually the default built-in camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # 2. Fetch the webcam's native width, height, and frames per second (FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Fallback FPS if the webcam doesn't report it properly
    if fps == 0 or fps > 100:
        fps = 30.0

    print(f"Recording Details: {frame_width}x{frame_height} at {fps} FPS")
    print("Press 'q' to stop recording and save the video.")

    # 3. Define the MP4 codec and initialize the VideoWriter
    # 'mp4v' is widely compatible for generating standard MP4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = 'output.mp4'
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        # Write the frame into the file
        out.write(frame)

        # Display the live preview window
        cv2.imshow('Recording Video... (Press Q to Quit)', frame)

        # Stop recording when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Clean up and release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video successfully saved as {output_filename}")

if __name__ == "__main__":
    capture_video()
