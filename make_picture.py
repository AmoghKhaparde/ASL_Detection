def add_pics() :
    import cv2
    import keyboard
    import os
    # import time

    # Define the destination folder
    destination_folder = input("What is the name of the folder?: ")
    destination_path = "pre_cropped_testing_pics/" + destination_folder

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    # Determine the starting image number based on existing files in the destination folder
    existing_files = os.listdir(destination_path)
    image_count = len([filename for filename in existing_files if filename.startswith("Photo-") and filename.endswith(".jpeg")])
    image_count += 1  # Increment to start from the next number

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            cv2.imshow('video feed', frame)
            cv2.setWindowProperty('video feed', cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(50)

            if keyboard.is_pressed("ctrl"):
                print("Adding new image:", image_count)
                image_filename = os.path.join(destination_path, "Photo-" + str(image_count) + ".jpeg")
                cv2.imwrite(image_filename, frame)
                image_count += 1  # Increment the image count
                # time.sleep(0.2)
            if keyboard.is_pressed("shift"):
                cap.release()
                cv2.destroyAllWindows()
                break
            if keyboard.is_pressed('backspace') and image_count != 0 :
                print('Deleting latest image')
                image_count -= 1
                os.remove(destination_path+"/Photo-" + str(image_count) + ".jpeg")
                # time.sleep(0.5)

    else:
        raise ConnectionError("Camera Not Found.")

    print("Images added: ", image_count - 1)
    print('\nDone!')

add_pics()
