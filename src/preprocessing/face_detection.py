# --- Face Detection Module ---
# Description: This module uses MediaPipe to detect faces in images.
# It provides functions to detect faces, crop the detected face, normalize coordinates,
# and visualize the detections with bounding boxes and keypoints.

# Libraries
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe Face Detection model
model_path = "models/mediapipe/blaze_face_short_range.tflite"

# MediaPipe options and model initialization
BaseOptions = mp.tasks.BaseOptions
FaceDetector = vision.FaceDetector
FaceDetectorOptions = vision.FaceDetectorOptions
VisionRunningMode = vision.RunningMode

# Setup the options for the FaceDetector
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),  # Path to the model
    running_mode=VisionRunningMode.IMAGE,                   # Running mode for image processing
    min_detection_confidence=0.5,                           # Minimum confidence for detection to be considered valid
    min_suppression_threshold=0.3,                          # Threshold for suppressing overlapping detections
)

# Create the FaceDetector with the specified options
face_detector = FaceDetector.create_from_options(options)


def detect_faces(image):
    """
    Detects faces in an image using MediaPipe Face Detection.
    Args:
        image (np.ndarray): Input image in BGR format.
    Returns:
        FaceDetectorResult: Result containing detected faces, or None if no faces are detected.
    """
    # Convert BGR to RGB for MediaPipe processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    # Perform face detection
    result = face_detector.detect(mp_image)
    # Check if any faces were detected
    if not result.detections:
        return None
    
    # Return the result containing detected faces
    return result


def crop_face(image, detection_result):
    """
    Crops the face from the image based on detection results.
    Args:
        image (np.ndarray): Input image in BGR format.
        detection_result (FaceDetectorResult): Result containing detected faces.
    Returns:
        List[np.ndarray]: List of cropped face images.
    """
    # Validate detection result
    if not detection_result or not detection_result.detections:
        return None
    
    # Extract the first detected face
    detection = detection_result.detections[0]
    # Get bounding box coordinates from the detection
    bbox = detection.bounding_box
    x, y, width, height = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
    # Get the image dimensions
    h_img, w_img, _ = image.shape
    # Calculate the coordinates for cropping
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x + width, w_img)
    y2 = min(y + height, h_img)
    # Crop the face from the image
    cropped_face = image[y1:y2, x1:x2]
    # Check if the cropped face is valid
    if cropped_face.size == 0:
        return None
    # Resize the cropped face to a fixed size (e.g., 48x48 pixels)
    cropped_face = cv2.resize(cropped_face, (48, 48), interpolation=cv2.INTER_LINEAR)
    # Return the cropped face image
    return cropped_face


def normalize_to_pixel_coordinates(norm_x, norm_y, width, height):
    """
    Converts normalized coordinates to pixel coordinates.
    
    Args:
        norm_x (float): Normalized x-coordinate (0 to 1).
        norm_y (float): Normalized y-coordinate (0 to 1).
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        Tuple[int, int]: Pixel coordinates (x, y).
    """
    # Check if the float value is between 0 and 1
    def is_valid_normalized_value(value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))
    
    # Validate normalized coordinates
    if not (is_valid_normalized_value(norm_x) and is_valid_normalized_value(norm_y)):
        return None
    
    # Convert normalized coordinates to pixel coordinates
    x_px = min(math.floor(norm_x * width), width - 1)
    y_px = min(math.floor(norm_y * height), height - 1)

    # Return the pixel coordinates as a tuple
    return x_px, y_px



def visualize_detections(image, detection_result):
    """
    Draws bounding boxes and keypoints on the image for detected faces.

    Args:
        image (np.ndarray): Input image in BGR format.
        detection_result (FaceDetectorResult): Result containing detected faces.

    Returns:
        np.ndarray: Image with visualized detections.
    """
    # Visualization parameters for bounding boxes and keypoints
    margin = 1  # pixels
    row_size = 1  # pixels
    font_size = 1
    font_thickness = 1
    text_color = (255, 0, 0)  # Blue color for text
    color = (0, 255, 0)  # Green color for keypoints
    thickness = 1  # Thickness for keypoints
    radius = 1  # Radius for keypoints

    # Create a copy of the image for annotation
    annotated_image = image.copy()
    # Extract image dimensions
    height, width, _ = image.shape

    # Iterate through each detection in the result
    for detection in detection_result.detections:
        # Get the bounding box coordinates
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        # Draw the rectangle on the image
        cv2.rectangle(annotated_image, start_point, end_point, text_color, 1)

        # Draw keypoints
        for keypoint in detection.keypoints:
            # Normalize keypoint coordinates to pixel coordinates
            keypoint_px = normalize_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            # Draw the keypoint circle on the image
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    result_text = category_name
    text_location = (margin + bbox.origin_x, margin + row_size + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

    return annotated_image