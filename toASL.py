import cv2
import imageio
import os
import time

sentence = input("What do you want to look up in the fingerspelling dictionary?: ")

# try :
sentence2 = sentence.casefold()
combined_frames = []

for letter in sentence2:
    gif1_path = "alphabet/"+letter+".gif"  # Replace with your actual file path
    with imageio.get_reader(gif1_path) as gif1_reader:
        frames = [frame for frame in gif1_reader]
        combined_frames.append(frames)

# Save the combined frames as a new GIF
combined_name = sentence2
combined_gif_path = "alphabet/" + combined_name + ".gif"
print("Saving Combined GIF file")
with imageio.get_writer(combined_gif_path, mode="I") as writer:
    for idx, frame in enumerate(combined_frames):
        print("Adding frame to Combined GIF file:", idx + 1)
        writer.append_data(frame)

print("Playing GIF in 3")
time.sleep(1)
print("Playing GIF in 2")
time.sleep(1)
print("Playing GIF in 1")
time.sleep(1)
print("Playing GIF:")

gif = cv2.VideoCapture(combined_gif_path)
cv2.namedWindow(sentence2)
cv2.namedWindow(sentence2)
cv2.moveWindow(sentence2, 450, 280)
cv2.setWindowProperty(sentence2, cv2.WND_PROP_TOPMOST, 1)
while True:
    ret, frame = gif.read()
    if not ret:
        break
    cv2.imshow(sentence2, frame)
    # Wait for a short duration and exit on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
# Release resources
gif.release()
cv2.destroyWindow(sentence2)
os.remove(combined_gif_path)

print('\nDone!')