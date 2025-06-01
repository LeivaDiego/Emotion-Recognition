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
import logging
from utils.logger import get_logger

# Setup logging
logger = get_logger(__name__)


class FaceDetectorMP:
    """
    A class to handle face detection using MediaPipe.
    It initializes the MediaPipe FaceDetector and provides methods for face detection and cropping.
    """
    # MediaPipe options and model initialization
    def __init__(self, mode="IMAGE", 
                 model_path="models/mediapipe/blaze_face_short_range.tflite",
                 verbose=False):
        """
        Initializes the MediaPipe FaceDetector with specified options.
        
        Args:
            mode (str): Running mode for the face detector. Default is "IMAGE".
            model_path (str): Path to the MediaPipe model file.
            verbose (bool): If True, enables detailed logging. Default is False.

        Raises:
            ValueError: If the model path is invalid or if the mode is not recognized.
            AttributeError: If the mode is not a valid MediaPipe RunningMode.
        """
        if verbose:
            # Enable Logging
            logger.setLevel(logging.DEBUG)
        else:
            # Disable Logging
            logger.setLevel(logging.ERROR)

        # MediaPipe options setup
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceDetector = vision.FaceDetector
        self.FaceDetectorOptions = vision.FaceDetectorOptions
        self.VisionRunningMode = vision.RunningMode
        mode = mode.upper()  # Convert mode to uppercase for consistency
        # Attribute to store the running mode
        self.running_mode = None

        # Initialize MediaPipe FaceDetector
        # Validate the model path
        if not model_path or not isinstance(model_path, str):
            logger.error("Invalid model path provided. Please provide a valid path to the model file.")
            raise ValueError("Invalid model path.")
        
        # Validate the mode
        # Check if the mode is a valid MediaPipe RunningMode
        if mode not in self.VisionRunningMode.__members__:
            logger.error(f"Invalid mode '{mode}'. Available modes: {', '.join(self.VisionRunningMode.__members__)}")
            raise AttributeError(f"Invalid mode '{mode}'.")
        
        else:
            logger.info(f"Initializing MediaPipe FaceDetector with model: {model_path}")
            
            if mode == "LIVE_STREAM":
                logger.warning("LIVE_STREAM mode is not supported in this implementation. Defaulting to IMAGE mode.")
                # Set the running mode to IMAGE by default
                mode = "IMAGE"
            
            self.running_mode = self.VisionRunningMode[mode.upper()]
            logger.info(f"Running mode set to: {self.running_mode}")
            
        # Setup the options for the FaceDetector
        options = self.FaceDetectorOptions(
            base_options=self.BaseOptions(model_asset_path=model_path), # Path to the model
            running_mode=self.running_mode,                             # Running mode for image processing
            min_detection_confidence=0.5,                               # Minimum confidence for detection to be considered valid
            min_suppression_threshold=0.3,                              # Threshold for suppressing overlapping detections
        )

        # Create the FaceDetector with the specified options
        self.face_detector = self.FaceDetector.create_from_options(options)
        logger.info("MediaPipe FaceDetector initialized successfully.")


    def detect_faces(self, image, timestamp=None):
        """
        Detects faces in an image using MediaPipe Face Detection.
        Args:
            image (np.ndarray): Input image in BGR format.
            timestamp (int, optional): Timestamp for the image. Default is None.
        Returns:
            FaceDetectorResult: Result containing detected faces, or None if no faces are detected.
        """
        # Convert BGR to RGB for MediaPipe processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        logger.debug("Starting face detection...")
        # Check if running mode is VIDEO
        # This requires a timestamp to be provided
        if self.running_mode == self.VisionRunningMode.VIDEO:
            # Validate the timestamp
            if timestamp is None:
                logger.error("Timestamp must be provided for VIDEO mode.")
                raise AttributeError("Timestamp not provided for VIDEO mode.")
            # Create a MediaPipe Timestamp object
            mp_timestamp = mp.Timestamp(seconds=timestamp)
            # Detect faces from the video frame
            result = self.face_detector.detect_for_video(mp_image, mp_timestamp)

        # Check if running mode is IMAGE
        # This does not require a timestamp
        elif self.running_mode == self.VisionRunningMode.IMAGE:
            # Detect faces from the image
            result = self.face_detector.detect(mp_image)

        else:
            logger.error(f"Unsupported running mode: {self.face_detector.running_mode}")
            raise ValueError(f"Unsupported running mode")
        
        # Check if any faces were detected
        if not result.detections:
            logger.warning("No faces detected in the image.")
            return None
        logger.debug(f"Detected {len(result.detections)} face(s) in the image.")

        # Return the result containing detected faces
        return result


    def crop_face(self, image, detection_result):
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
            logger.warning("No faces detected in the image.")
            return None
        
        logger.debug("Cropping the first detected face from the image...")
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
        # Log the cropping coordinates
        logger.debug(f"Cropping coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        # Check if the cropped face is valid
        if cropped_face.size == 0:
            logger.warning("Cropped face is empty. No valid face detected.")
            return None
        logger.debug("Face cropped successfully.")
        # Return the cropped face image
        return cropped_face


    def normalize_to_pixel_coordinates(self, norm_x, norm_y, width, height):
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



    def annotate_face_detections(self, image, detection_result):
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
        logger.debug("Annotating face detections on the image...")
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
                keypoint_px = self.normalize_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
                # Draw the keypoint circle on the image
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
        
        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        result_text = category_name
        text_location = (margin + bbox.origin_x, margin + row_size + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
        
        logger.debug("Face detections annotated successfully.")
        
        # Return the annotated image with bounding boxes and keypoints
        return annotated_image