from ultralytics import YOLO
import cv2

# load the yolov8 model

model = YOLO("yolov8x.pt")

# specify how many frames are going to be reviewed (lower number will increase time, need to find minimum that can provide accurate result)
SEQUENCE_LENGTH = 20

# frame window = number of frames from end of the clip to be reviewed, idea is that only the last 3 seconds are actually important which translates to roughly 120 frames depending on rate
FRAME_WINDOW = 120

# load the video

video_path = '/Users/mfitzpatrick/Documents/Data Science/Skateboard Model/skateboard_model/Version 2/data/avi_files/Bail/B32.avi'
cap = cv2.VideoCapture(video_path)

#calculate frames in video
video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#calculate the start frame to analyse
start_frame = max(int(video_frames_count - FRAME_WINDOW), 1)
#calculate the number of frames to use
frames_used = int(video_frames_count - start_frame)
#calculate the number of frames to skip in order to get a sequence of SEQUENCE_LENGTH that is evenly spaced
skip_frames_window = max(int(frames_used / SEQUENCE_LENGTH), 1)

# Create/open a text file for writing bounding box data
output_file_path = 'bounding_box_data.txt'
output_file = open(output_file_path, 'w')

ret = True
# read frames
for frame_counter in range(SEQUENCE_LENGTH):

    print(f'extracting frames for {video_path} on {frame_counter}')

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_counter * skip_frames_window)

    success, frame = cap.read()

    if not ret:
        break

    # detect and track objects
    result = model(frame)

    boxes = result[0].boxes.xyxy.tolist()
    classes = result[0].boxes.cls.tolist()
    names = result[0].names

    # Initialize varibales to store the largest bounding box for each class
    largest_person_box = None
    largest_skateboard_box = None
    frame_info_string = ''

    # Iterate through the results and write information to the text file
    for box, cls in zip(boxes, classes):

        detected_class = cls
        name = names[int(cls)]

        # Check if the detected class is a 'person' or 'skateboard'
        if name in ['person', 'skateboard']:
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)

            # Update the largest bounding box for the corresponding class
            if name == 'person' and (largest_person_box is None or box_area > largest_person_box[0]):
                largest_person_box = (box_area, box)
            elif name == 'skateboard' and (largest_skateboard_box is None or box_area > largest_skateboard_box[0]):
                largest_skateboard_box = (box_area, box)

    # Write information for the largest bounding box of each class to the output file
    if largest_person_box:
        x1, y1, x2, y2 = largest_person_box[1]
        frame_info_string += f"Frame: {frame_counter}, Class: person, Bounding Box: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}) | "
    if largest_skateboard_box:
        x1, y1, x2, y2 = largest_skateboard_box[1]
        frame_info_string += f"Frame: {frame_counter}, Class: skateboard, Bounding Box: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}) | "

    output_file.write(frame_info_string)



# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()