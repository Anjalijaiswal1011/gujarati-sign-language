import cv2
import os
import sys

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "raw_images")

def main():
    print("=== Custom Gesture Data Collection ===")
    label = input("Enter the gesture label (e.g., A, B, Hello, Thanks): ").strip()
    if not label:
        print("Label cannot be empty.")
        sys.exit(1)

    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    # Get the current number of images to avoid overwriting
    existing_images = len([f for f in os.listdir(label_dir) if f.endswith('.jpg')])
    count = existing_images

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print(f"Saving to {label_dir}")
    print("Press 's' to save an image.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for easier capturing
        frame = cv2.flip(frame, 1)

        # Draw a box indicating the capture area
        h, w, _ = frame.shape
        box_size = 300
        start_x = w // 2 - box_size // 2
        start_y = h // 2 - box_size // 2
        
        cv2.rectangle(frame, (start_x, start_y), (start_x + box_size, start_y + box_size), (0, 255, 0), 2)
        cv2.putText(frame, f"Label: {label} | Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the ROI
            roi = frame[start_y:start_y + box_size, start_x:start_x + box_size]
            filename = os.path.join(label_dir, f"{count}.jpg")
            cv2.imwrite(filename, roi)
            print(f"Saved {filename}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
