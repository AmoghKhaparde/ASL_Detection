#fixing the model starts at 27, 27 is 
#32 is back to resnet 18

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import mediapipe as mp
import statistics  # Added for mode calculation
from time import time, sleep
import speakify
import pygame
import requests
import urllib.request

pygame.init()

class_labels = [chr(i).upper() for i in range(ord('a'), ord('z')+1)]
num_classes = len(class_labels)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.001)
mp_drawing = mp.solutions.drawing_utils

# Load your trained model's weights
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Load saved weights
model.load_state_dict(torch.load('data/asl_model26.pth'))
model.eval()

# Move the model to the appropriate device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define data transformations for testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=1.5, contrast=1.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Open a video capture source (0 for webcam)
video_capture = cv2.VideoCapture(0)
sentence = " "
num_predictions_per_sec = 10  # Adjust as needed

def make_prediction(input_tensor):
    with torch.no_grad():
        # Make prediction
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        # print(probabilities)
        max_prob, predicted = torch.max(probabilities, 1)
        # Get class label and probability
        predicted_label = class_labels[predicted.item()]
        predicted_prob = max_prob.item()
    return [predicted_label, predicted_prob]

n = 0
text = ''
prediction_buffer = []
while True:
    # Capture a frame
    ret, frame = video_capture.read()
    if not ret:
        break

    results = hands.process(frame)

    frame_height, frame_width, _ = frame.shape
    text_size = cv2.getTextSize(sentence, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = frame_height - 10  # A little above the bottom edge

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            x_min = x_min - 30
            x_max = x_max + 15
            y_min = y_min - 15
            y_max = y_max + 30


        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        hand_frame = frame[y_min:y_max, x_min:x_max]
        cv2.imshow('model input', hand_frame)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        pil_frame = Image.fromarray(hand_frame)
        input_tensor = transform(pil_frame).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        predicted_label, predicted_prob = make_prediction(input_tensor)

        prediction_buffer.append(predicted_label)

        if len(prediction_buffer) == num_predictions_per_sec:
            predicted_label = statistics.mode(prediction_buffer)
            print(prediction_buffer)
            prediction_buffer = []

            text = 'Predicted: {a} (Accuracy: {b}%)'.format(a=predicted_label, b=(str(predicted_prob * 100)[0:4]))
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if predicted_prob >= 0.45:
                if predicted_label == sentence[-1]:
                    if n == 20:  # change this number depending on fingerspelling speed
                        n = 0
                        sentence += predicted_label
                        frame_height, frame_width, _ = frame.shape
                        text_size = cv2.getTextSize(sentence, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
                        text_x = (frame_width - text_size[0]) // 2
                        text_y = frame_height - 10  # A little above the bottom edge
                        cv2.putText(frame, sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    else:
                        n += 1
                else:
                    n = 0
                    frame_height, frame_width, _ = frame.shape
                    text_size = cv2.getTextSize(sentence, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
                    text_x = (frame_width - text_size[0]) // 2
                    text_y = frame_height - 10  # A little above the bottom edge
                if predicted_label != " " and predicted_label != sentence[-1]:
                    sentence += predicted_label
                cv2.putText(frame, sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    else:
        cv2.putText(frame, "No Hand Detected (space)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        if sentence[-1] != ' ':
            sentence += ' '

    key = cv2.waitKey(1)  # asyncio
    try:
        if key == ord('q'):
            break
        if key == 8:
            if len(sentence) > 0:
                sentence = sentence[:-2]
                if sentence == '':
                    sentence += ' '
                    cv2.putText(frame, sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if key == 32:
            sentence += ' '
            cv2.putText(frame, sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if key == 13 :
            x = speakify.text_to_voice(sentence)
            # MP3 file URL
            mp3_url = x
            # Download the MP3 file
            mp3_file, headers = urllib.request.urlretrieve(mp3_url)
            # Initialize the mixer
            pygame.mixer.init()
            # Load the MP3 file
            pygame.mixer.music.load(mp3_file)
            # Play the MP3 file
            pygame.mixer.music.play()
            # Wait for the music to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(1)
            # Clean up
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            sentence = ' '
            print("If no audio is audible try increasing volume or clicking on this link instead: "+ x)
    except:
        pass

    cv2.putText(frame, sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow('ASL Detection', frame)
    cv2.setWindowProperty('ASL Detection', cv2.WND_PROP_TOPMOST, 1)

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()

bounding box models start at model 18, but 18-19 dont work as well, so try starting from 20
full static cnn based model is at model 12
