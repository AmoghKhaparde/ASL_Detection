import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import mediapipe as mp
from time import time,sleep
# 0 class is hand is present but not legible
class_labels = []
for i in 'abcd' :
    class_labels.append(i.upper())
num_classes = len(class_labels)

hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.1) # make sure the confidence of seeing a hand is above 0.5
mp_drawing = mp.solutions.drawing_utils

# Load your trained model's weights
model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Load saved weights
model.load_state_dict(torch.load('data/asl_model38.pth'))
model.eval()

# Move the model to the appropriate device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define data transformations for testing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Open a video capture source (0 for webcam)
video_capture = cv2.VideoCapture(0)
sentence = " "
# num_predictions_sec = int(input("Choose a number from 1-10 for how many predictions made in a second: "))

def make_prediction(input_tensor):
    with torch.no_grad():
        # Make prediction
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
        # Get class label and probability
        print(predicted.item())
        predicted_label = class_labels[predicted.item()]
        predicted_prob = max_prob.item()
    return [predicted_label, predicted_prob]

repeated_predictions = 0 # this is part of a system that requires you to hold a sign for a long time if you want it to repeat
five_counts = 0
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

    if results.multi_hand_landmarks: # checks if a hand is even visible in the camera
        # Get bounding box coordinates around the hand
        # for hand_landmarks in results.multi_hand_landmarks:
        #     x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        #     for landmark in hand_landmarks.landmark:
        #         x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        #         x_min = min(x_min, x)
        #         y_min = min(y_min, y)
        #         x_max = max(x_max, x)
        #         y_max = max(y_max, y)
        
        #     x_min = x_min-30
        #     x_max = x_max+15
        #     y_min = y_min-15
        #     y_max = y_max+30

        # Crop the frame to include only the hand region
        # hand_frame = frame[y_min:y_max, x_min:x_max]
        # cv2.imshow('model input', hand_frame)
        # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Convert OpenCV frame to PIL Image
        pil_frame = Image.fromarray(frame) # switch this back to variable hand_frame
        # Apply transformations
        input_tensor = transform(pil_frame).unsqueeze(0)
        # Move input tensor to the same device as the model
        input_tensor = input_tensor.to(device)
        # Get predictions
        predicted_label, predicted_prob = make_prediction(input_tensor)

        text = 'Predicted: {a} (Probability: {b}%)'.format(a=predicted_label, b=(str(predicted_prob * 100)[0:4]))
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if predicted_prob >= 0.6: # making predicted probability is higher than 0.6
            # in the next section, it checks the sentence for the pervious character and compared it to the predicted label. 
            # if both are the same, then it waits a bit (20 iterations of predictions) and if all of those are the same
            # it will finally add that prediction to the sentence.
            # if it is not a repeat of the previous character, just add the predicted label to the sentence
            if predicted_label == sentence[-1]: # make sure that previous character is not repeat many many times
                if repeated_predictions == 20:  # change this number depending on fingerspelling speed
                    repeated_predictions = 0
                    sentence += predicted_label
                    frame_height, frame_width, _ = frame.shape
                    # Get the size of the text
                    text_size = cv2.getTextSize(sentence, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
                    # Calculate the position to center the text at the bottom middle
                    text_x = (frame_width - text_size[0]) // 2
                    text_y = frame_height - 10  # A little above the bottom edge
                    cv2.putText(frame, sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                else:
                    repeated_predictions += 1
            else:
                repeated_predictions = 0
                frame_height, frame_width, _ = frame.shape

                # Get the size of the text
                text_size = cv2.getTextSize(sentence, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]

                # Calculate the position to center the text at the bottom middle
                text_x = (frame_width - text_size[0]) // 2
                text_y = frame_height - 10  # A little above the bottom edge
                
                # if five_counts == 5 :
                #     five_counts = 0
                sentence += predicted_label
            cv2.putText(frame, sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

# do the delay thing for 

    else:
        cv2.putText(frame, "No Hand Detected (space)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        if sentence[-1] != ' ':
            sentence += ' '

    key = cv2.waitKey(100) # asyncio
    # Break loop on pressing 'q'
    try:
        if key == ord('q'):
            break
        if key == 8: # 8 = backspace
            # simply allows user to change sentence variable before submitting it to speakify portion of the system
            if len(sentence) > 0:
                sentence = sentence[:-1]
                if sentence == '':
                    sentence += ' '
                cv2.putText(frame, sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if key == 32: # 32 = space
            sentence += ' ' # this section adds space to the sentence to separate letters
            cv2.putText(frame, sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    except:
        pass

    cv2.putText(frame, sentence, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    # Display the frame
    cv2.imshow('ASL Detection', frame)
    cv2.setWindowProperty('ASL Detection', cv2.WND_PROP_TOPMOST, 1)

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
