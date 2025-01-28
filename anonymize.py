import cv2
import dlib
import argparse
import os

def blur_region(frame, x1, y1, x2, y2, ksize=(99, 99), sigma=30):
    """
    Helper function to blur a region in the given frame.
    (x1, y1) should be the top-left corner; (x2, y2) the bottom-right.
    """
    # Ensure coordinates are within the frame and sorted
    x1, x2 = sorted([max(0, x1), min(frame.shape[1], x2)])
    y1, y2 = sorted([max(0, y1), min(frame.shape[0], y2)])

    roi = frame[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, ksize, sigma)
    frame[y1:y2, x1:x2] = blurred_roi

def blur_faces_in_frame(frame, detector):
    """
    Detect and blur faces in a single frame using dlib's face detector.
    Returns the processed frame.
    """
    # Convert to RGB for dlib
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector(rgb_frame)

    # Blur each detected face
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        blur_region(frame, x, y, x + w, y + h)

    return frame

def blur_manual_boxes_in_frame(frame, boxes):
    """
    Blur any manual bounding boxes in the given frame.
    boxes is a list of (x, y, w, h) or (x1, y1, x2, y2).
    """
    for box in boxes:
        if len(box) == 4:
            # If box is (x, y, w, h), convert to corners
            x, y, w, h = box
            x1, y1, x2, y2 = x, y, x + w, y + h
            blur_region(frame, x1, y1, x2, y2)
        else:
            # If it's already corners (x1, y1, x2, y2)
            x1, y1, x2, y2 = box
            blur_region(frame, x1, y1, x2, y2)

def get_manual_boxes(frame):
    """
    Ask the user how many additional boxes to blur.
    Then let them select bounding boxes using OpenCV's ROI tool.
    Returns a list of bounding boxes in (x, y, w, h) format.
    """
    boxes = []
    num_boxes_str = input("Enter the number of boxes to manually blur: ")
    try:
        num_boxes = int(num_boxes_str)
    except ValueError:
        print("Invalid number. No boxes will be manually selected.")
        return boxes

    for i in range(num_boxes):
        print(f"\nSelect bounding box #{i+1}. Close the ROI window when done.")
        # cv2.selectROI returns (x, y, w, h)
        roi = cv2.selectROI("Select bounding box", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select bounding box")

        if roi == (0, 0, 0, 0):
            print("No selection made, skipping.")
        else:
            boxes.append(roi)

    return boxes

def blur_faces_in_image(input_path, output_path, detector, manual=False, only_manual=False):
    """
    Process a single image.
    - If only_manual=True, skip dlib face detection and only blur manually selected boxes.
    - If manual=True (and only_manual=False), do face detection + manual boxes.
    - Otherwise, do face detection only.
    """
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Unable to load image {input_path}")
        return

    processed_image = image.copy()

    if only_manual:
        print("Using ONLY manual blur boxes (no face detection).")
        manual_boxes = get_manual_boxes(processed_image)
        if manual_boxes:
            blur_manual_boxes_in_frame(processed_image, manual_boxes)
    else:
        # 1) Face detection
        processed_image = blur_faces_in_frame(processed_image, detector)

        # 2) If manual is set, also blur additional boxes
        if manual:
            print("Manual box selection enabled for this image.")
            manual_boxes = get_manual_boxes(processed_image)
            if manual_boxes:
                blur_manual_boxes_in_frame(processed_image, manual_boxes)

    # Save the output
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to {output_path}")

def blur_faces_in_video(input_path, output_path, detector, manual=False, only_manual=False):
    """
    Process a video.
    - If only_manual=True, skip dlib face detection altogether; 
      let user select boxes in the first frame, then apply those boxes to every frame.
    - If manual=True (and only_manual=False), do face detection + manual boxes in first frame.
    - Otherwise, do face detection only on every frame.
    """
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        print(f"Error: Unable to load video {input_path}")
        return

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    manual_boxes = []
    if only_manual or manual:
        # We need to get manual boxes from the first frame
        ret, first_frame = video.read()
        if not ret:
            print("Error: Could not read first frame of video.")
            return

        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

        print("Please select manual blur boxes on the first frame.")
        manual_boxes = get_manual_boxes(first_frame)

    frame_count = 0
    print("Processing video...")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if only_manual:
            # Only manual boxes, no face detection
            if manual_boxes:
                blur_manual_boxes_in_frame(frame, manual_boxes)
        else:
            # Face detection on the frame
            frame = blur_faces_in_frame(frame, detector)
            # If also in manual mode, apply manual boxes
            if manual:
                if manual_boxes:
                    blur_manual_boxes_in_frame(frame, manual_boxes)

        out.write(frame)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    video.release()
    out.release()
    print(f"Processed video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Blur faces in an image or video.")
    parser.add_argument("--input", required=True, help="Path to the input image or video.")
    parser.add_argument("--output", required=True, help="Path to the output image or video.")
    parser.add_argument("--manual", action="store_true",
                        help="Enable face detection + manual selection of additional boxes.")
    parser.add_argument("--only-manual", action="store_true",
                        help="Skip face detection and ONLY use manually drawn blur boxes.")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    manual_mode = args.manual
    only_manual_mode = args.only_manual

    # Check if the input file exists
    if not os.path.isfile(input_path):
        print(f"Error: File {input_path} does not exist.")
        return

    # Initialize dlib face detector
    detector = dlib.get_frontal_face_detector()

    # Check the file extension to see if it's an image or video
    is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png'))
    is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_image:
        print("Processing image...")
        blur_faces_in_image(input_path, output_path, detector,
                            manual=manual_mode,
                            only_manual=only_manual_mode)
    elif is_video:
        print("Processing video...")
        blur_faces_in_video(input_path, output_path, detector,
                            manual=manual_mode,
                            only_manual=only_manual_mode)
    else:
        print("Error: Unsupported file format. Please provide an image "
              "(.jpg, .jpeg, .png) or video (.mp4, .avi, .mov, .mkv).")

if __name__ == "__main__":
    main()
