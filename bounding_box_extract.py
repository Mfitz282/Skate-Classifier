from ultralytics import YOLO
import cv2
import numpy as np
import os

# load the yolov8 model

model = YOLO("yolov8x.pt")

# specify how many frames are going to be reviewed (lower number will increase time, need to find minimum that can provide accurate result)
SEQUENCE_LENGTH = 20

# frame window = number of frames from end of the clip to be reviewed, idea is that only the last 3 seconds are actually important which translates to roughly 120 frames depending on rate
FRAME_WINDOW = 120

video_directory = '/Users/mfitzpatrick/Pictures/GoPro/SkateModelClips/Testing Data/2022-09-24 Skate Romsey/HERO8 Black 1/'

allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.MP4']

# List files in the directory and filter by allowed extensions
video_list = [filename for filename in os.listdir(video_directory) if any(filename.endswith(ext) for ext in allowed_extensions)]

output_directory = video_directory


# load the video
for video_path in video_list:

    cap = cv2.VideoCapture(video_directory + video_path)

    #get the dimensions of the video - used later on to normalise the dimensions of the bounding boxes
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'{width}, {height}')

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

    person_position = []
    skateboard_position = []

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
            frame_info_string += f"Frame: {frame_counter}, Person: ({(x1/width):.3f}, {(y1/height):.3f}, {(x2/width):.3f}, {(y2/height):.3f}) "
            person_position.append([(x1/width), (y1/height), (x2/width), (y2/height)])

        else:
            person_position.append(np.zeros(4))

        if largest_skateboard_box:
            x1, y1, x2, y2 = largest_skateboard_box[1]
            frame_info_string += f"Skateboard: ({(x1/width):.3f}, {(y1/height):.3f}, {(x2/width):.3f}, {(y2/height):.3f}) "
            skateboard_position.append([(x1 / width), (y1 / height), (x2 / width), (y2 / height)])

        else:
            skateboard_position.append(np.zeros(4))

        #taken out this for now, was using it as originally was going to create a data table within a text file, now going to save the arrays
        #output_file.write(frame_info_string)

    # Convert lists to NumPy arrays
    person_position = np.array(person_position)
    skateboard_position = np.array(skateboard_position)

    # Stack the arrays to create a 3D array
    reshaped_data = np.stack([person_position, skateboard_position], axis=1)

    np.save(output_directory + video_path[:-4],reshaped_data,)

    cap.release()
    cv2.destroyAllWindows()