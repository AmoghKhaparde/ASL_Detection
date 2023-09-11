import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils

for alphabet in 'abcdefghijkl' :
    
    folder_path = 'test_images/'  # Replace with the path to the parent folder where you want to create the new folder
    # Join the parent folder path and the new folder name to get the full path
    new_folder_path = os.path.join(folder_path, alphabet)
    # Use the os.makedirs() function to create the folder
    os.makedirs(new_folder_path)

    for num in range(1, len(os.listdir('old_test_images/'+alphabet+'/'))+1) :
        try:
            # Path to your input JPEG image
            input_image_path = 'old_test_images/'+alphabet+'/Photo-'+str(num)+'.jpeg'

            # Load the input image
            image = cv2.imread(input_image_path)

            # Convert the image to RGB format
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect hands in the image
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                # Get bounding box coordinates around the hand
                for hand_landmarks in results.multi_hand_landmarks:
                    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)


                    # Crop the hand region from the original image
                    x_min = x_min-30
                    x_max = x_max+15
                    y_min = y_min-15
                    y_max = y_max+30
                    hand_crop = image[y_min:y_max, x_min:x_max]

                    # Path to the output folder
                    output_folder = 'test_images/'+alphabet

                    # Save the cropped hand image
                    cv2.imwrite(output_folder + '/Photo-'+str(num)+'.jpeg', hand_crop)
        except :
            continue

# Release resources
hands.close()