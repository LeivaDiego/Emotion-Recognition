# --- Face Landmark Detection Module ---
# Description: This module detects facial landmarks in images using the MediaPipe tasks.
# It provides functions to initialize the model, process images, and extract landmarks.

# Libraries
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
from utils.logger import get_logger

# Setup logging
logger = get_logger(__name__)

class LandmarkDetectorMP:
    """
    A clas to handle facial landmark detection using MediaPipe.
    It initializes the MediaPipe FaceLandmarker model and provides methods to process images
    """

    # MediaPipe options and model initialization
    def __init__(self, mode="IMAGE", 
                 model_path="models/mediapipe/face_landmarker.task",
                 resizing=True,
                 verbose=False):
        """
        Initializes the MediaPipe FaceLandmarker model with specified options.
        Args:
            mode (str): The running mode for the FaceLandmarker. Options are 'IMAGE', 'VIDEO', or 'LIVE_STREAM'.
            model_path (str): Path to the MediaPipe FaceLandmarker model file.
            resizing (bool): Flag to enable or disable resizing of the input image.
            verbose (bool): Flag to enable verbose logging.

        Raises:
            ValueError: If the model path is invalid or if the mode is not supported.
            AttributeError: If the MediaPipe tasks module is not properly initialized.
        """
        if verbose:
            # Enable logging
            logger.setLevel(logging.DEBUG)
        else:
            # Disable logging
            logger.setLevel(logging.ERROR)

        # MediaPipe Options setup
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        mode = mode.upper() # Ensure mode is uppercase

        # Resizing option
        self.enable_resizing = resizing

        # Initialize MediaPipe FaceLandmarker
        # Validate model path
        if not model_path or not isinstance(model_path, str):
            logger.error("Invalid model path provided. Please provide a valid path to the model file.")
            raise ValueError("Invalid model path.")
        
        # Validate the mode
        # Check if the mode is a valid MediaPipe RunningMode
        if mode not in self.VisionRunningMode.__members__:
            logger.error(f"Invalid mode '{mode}'. Available modes: {', '.join(self.VisionRunningMode.__members__)}")
            raise AttributeError(f"Invalid mode '{mode}'.")
        
        else:
            # Initialize the FaceLandmarker with the provided model path and mode
            logger.info(f"Initializing MediaPipe FaceLandmarker with model: {model_path}")
            # Check if the mode is LIVE_STREAM
            # Not supported in this implementation, so default to IMAGE mode
            if mode == "LIVE_STREAM":
                logger.warning("LIVE_STREAM mode is not supported in this implementation. Defaulting to IMAGE mode.")
                # Set the running mode to IMAGE by default
                mode = "IMAGE"
            # Set the running mode
            self.running_mode = self.VisionRunningMode[mode.upper()]
            logger.info(f"Running mode set to: {self.running_mode}")

        # Setup the options for the FaceLandmarker
        options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path), # Path to the model file
            running_mode=self.running_mode,                             # Set the running mode
            num_faces=1,                                                # Default to detecting one face
            min_face_detection_confidence=0.5,                          # Minimum detection confidence
            min_face_presence_confidence=0.5,                           # Minimum face presence confidence
            min_tracking_confidence=0.5,                                # Minimum tracking confidence
            output_face_blendshapes=False,                              # Disable blendshapes output
            output_facial_transformation_matrixes=False                 # Disable transformation matrices output
        )

        # Create the FaceLandmarker instance with the specified options
        self.face_landmarker = self.FaceLandmarker.create_from_options(options)
        logger.info("MediaPipe FaceLandmarker initialized successfully.")


    def detect_landmarks(self, image, timestamp=None):
        """
        Detects facial landmarks in the provided image.
        Args:
            image (numpy.ndarray): The input image in BGR format.
            timestamp (int, optional): Timestamp for the image, if applicable.

        Returns:
            list: A list of detected landmarks, each represented as a list of (x, y) coordinates.
        """
        # Convert BGR to RGB for MediaPipe processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image if resizing is enabled
        if self.enable_resizing:
            logger.debug("Resizing the cropped face to a fixed size (128x128 pixels).")
            # Resize the cropped face to a fixed size
            rgb_image = cv2.resize(rgb_image, (128, 128), interpolation=cv2.INTER_LINEAR)

        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        logger.debug("Starting landmark detection...")
        # Check if the running mode is VIDEO
        # This requires a timestamp to be provided
        if self.running_mode == self.VisionRunningMode.VIDEO:
            # Validate the timestamp
            if timestamp is None:
                logger.error("Timestamp must be provided for VIDEO mode.")
                raise AttributeError("Timestamp not provided for VIDEO mode.")
            # Detect faces from the video frame
            result = self.face_landmarker.detect_for_video(mp_image, timestamp)

        # Check if running mode is IMAGE
        # This does not require a timestamp
        elif self.running_mode == self.VisionRunningMode.IMAGE:
            # Detect faces from the image
            result = self.face_landmarker.detect(mp_image)

        else:
            logger.error(f"Unsupported running mode: {self.face_landmarker.running_mode}")
            raise ValueError(f"Unsupported running mode")
        
        # Check if any landmarks were detected
        if not result.face_landmarks:
            logger.warning("No face landmarks detected.")
            return None
        logger.debug(f"Detected {len(result.face_landmarks)} face with landmarks.")

        # Return the landmarks
        return result
    

    def extract_landmark_vector(self, result):
        """
        Flattens the 3D landmarks of the first detected face into a 1D vector (478 x 3 = 1434).
        
        Args:
            result: FaceLandmarkerResult from detect_landmarks().
        
        Returns:
            np.ndarray of shape (1434,) or None if no landmarks found.
        """
        if not result or not result.face_landmarks:
            logger.warning("No face landmarks found in the result.")
            return None
        logger.debug("Extracting 3D landmarks from the first detected face.")
        # Extract the first detected face landmarks
        landmarks = result.face_landmarks[0]
        # Flatten the landmarks into a 1D vector
        # Each landmark has x, y, z coordinates, so we create a flat list
        vector = [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]

        # Convert the list to a numpy array of type float32
        vector_1d = np.array(vector, dtype=np.float32)
        logger.debug(f"Extracted landmark vector of shape: {vector_1d.shape}")
        # Return the flattened vector
        return vector_1d



    def annotate_face_landmarks(self, image, landmarks):
        """
        Draws the detected face landmarks on the image for the detected face.

        Args:
            image (numpy.ndarray): The input image in BGR format.
            landmarks (mediapipe.tasks.python.vision.FaceLandmarkerResult): The result containing detected landmarks.

        Returns:
            numpy.ndarray: The annotated image with landmarks drawn.
        """
        logger.debug("Annotating face landmarks on the image...")
        # Check if landmarks are provided
        face_landmarks_list = landmarks.face_landmarks
        annotated_image = np.copy(image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])
            logger.debug(f"Drawing landmarks for face {idx + 1} with {len(face_landmarks_proto.landmark)} landmarks.")
            # Draw the face mesh tesselation, contours, and irises.
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            logger.debug(f"Drawing face mesh tesselation for face {idx + 1}.")
            # Draw the face mesh contours.
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            logger.debug(f"Drawing face mesh contours for face {idx + 1}.")
            # Draw the face mesh irises.
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())
        
        logger.debug("Face landmarks annotation completed successfully.")
        # Return the annotated image with landmarks drawn
        return annotated_image