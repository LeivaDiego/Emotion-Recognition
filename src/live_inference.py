# --- Live Emotion Recognition using MLP Model and MediaPipe ---
# This script captures video from the webcam, detects faces, extracts landmarks,
# and predicts emotions using a pre-trained MLP model.

# Libraries
# Standard libraries
import cv2  
import torch

# Custom libraries
from models.mlp_model import EmotionClassifier
from preprocessing.face_detection import FaceDetectorMP
from preprocessing.landmark_detection import LandmarkDetectorMP
from utils.logger import get_logger

# Set up logger
logger = get_logger(__name__)

# CUDA device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device Detected: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
logger.info(f"Using device: {device}")

# Labels and color mapping for emotions
label_map = {
    0: "anger",     
    1: "happiness", 
    2: "neutral", 
    3: "sadness", 
    4: "surprise"
    }

color_map = {
    "anger": (0, 0, 255),       # Red
    "happiness": (0, 255, 0),   # Green
    "neutral": (100, 100, 100), # GRAY
    "sadness": (255, 0, 0),     # Blue
    "surprise": (0, 255, 255)   # Yellow
}
# - - Initialization and Model Loading - -
try:
    # Model initialization and loading
    model = EmotionClassifier().to(device)
    logger.info("Initialized EmotionClassifier model.")
    # Load the pre-trained model weights
    model.load_state_dict(torch.load("src/models/pytorch/emotion_model.pth", map_location=device))
    logger.info("Loaded pre-trained model weights.")
    model.eval()

    # Initialize MediaPipe face and landmark detectors in video mode
    face_detector = FaceDetectorMP(mode="VIDEO", model_path="src/models/mediapipe/blaze_face_short_range.tflite")
    logger.info(f"Initialized MediaPipe FaceDetector with model: blaze_face_short_range.tflite")

    landmark_detector = LandmarkDetectorMP(mode="VIDEO", model_path="src/models/mediapipe/face_landmarker.task")
    logger.info(f"Initialized MediaPipe LandmarkDetector with model: face_landmarker.task")

except Exception as e:
    logger.error(f"An error occurred during model initialization: {e}")
    raise RuntimeError("Failed to initialize model or detectors.")

# - - Live Inference Loop - -
try:
    # Video capture setup
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Could not open webcam. Please check the camera connection.")
        raise RuntimeError("Webcam not accessible.")

    logger.info("Starting video capture from webcam.")
    # Main loop for live inference
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to capture frame from webcam. Stoping...")
            break
        
        # Get the current timestamp in milliseconds
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Step 1: Detect faces in the current frame
        detection_result = face_detector.detect_faces(frame, timestamp)
        # If no faces are detected, continue to the next frame
        if detection_result is None:
            continue

        # Step 2: Crop the face region from the frame
        # If no cropped face is found, continue to the next frame
        # This step is crucial to ensure we only process the face region
        cropped = face_detector.crop_face(frame, detection_result)
        if cropped is None:
            continue

        # Step 3: Detect landmarks in the cropped face
        # If no landmarks are detected, continue to the next frame
        landmark_result = landmark_detector.detect_landmarks(cropped, timestamp)
        if landmark_result is None:
            continue

        # Step 4: Extract landmark vector from the landmark result
        # If the landmark vector is None, continue to the next frame
        vector = landmark_detector.extract_landmark_vector(landmark_result)
        if vector is None:
            continue

        # Step 5: Prepare the input tensor and perform inference
        # Ensure the vector is a 1D array and convert it to a tensor
        input_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(device)
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Perform inference using the MLP model
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()
            label = label_map[pred]
            color = color_map[label]

        # Step 6: Draw bounding box and label on the frame
        # Ensure the detection result has at least one detection
        bbox = detection_result.detections[0].bounding_box
        x1, y1 = bbox.origin_x, bbox.origin_y
        x2, y2 = x1 + bbox.width, y1 + bbox.height

        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Step 7: Display the frame with the bounding box and label
        cv2.imshow("Emotion Recognition", frame)

        # exit condition
        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    logger.info("Live inference interrupted by user.")

except Exception as e:
    logger.error(f"An error occurred during live inference: {e}")
    
finally:
    # Release the video capture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Released video capture and destroyed all OpenCV windows.")
    # End of live inference script