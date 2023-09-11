import cv2
import imageio
import keyboard

cap = cv2.VideoCapture(0)
frames = []
image_count = 0
if cap.isOpened() :
    while True:
        ret, frame = cap.read()
        image_count += 1
        frames.append(frame)
        print("Adding new image:", image_count)
        if keyboard.is_pressed("shift"):
            cap.release()
            cv2.destroyAllWindows()
            break
else :
    raise ConnectionError("Camera Not Found.")


print("Images added: ", len(frames))
name = input("What is the name of the gif file?: ")
print("Saving GIF file")
with imageio.get_writer("alphabet/"+name+".gif", mode="I") as writer:
    for idx, frame in enumerate(frames):
        print("Adding frame to GIF file: ", idx + 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

print('\nDone!')