import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (You can replace "yolov8n.pt" with the path to your model)
model = YOLO('yolov8n.pt')

# Open the video file
camera = cv2.VideoCapture("car1.mp4")

# Loop through video frames
while True:
    suc, frame = camera.read()

    if not suc:
        break  # Exit if there are no more frames

    # Save the frame as an image
    cv2.imwrite("frame_image.png", frame)

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Loop through detection results and draw rectangles around detected objects
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
        print(x1,y1,x2,y2)

    # Display the frame with bounding boxes
    cv2.imshow("Car Detection", cv2.resize(frame,(640,480)))

    # Wait for 1 ms and check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # break

# Release resources
camera.release()
cv2.destroyAllWindows()