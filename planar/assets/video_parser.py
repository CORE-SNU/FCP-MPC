import cv2
import os
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("asset_dir", help="asset dirpath to save", type=str)
args = parser.parse_args()


desired_fps = 2.5
interval = 1.0 / desired_fps  

video_files = glob.glob(os.path.join(os.path.dirname(__file__), "videos", "*.avi"))


for video_file in video_files:
    print("Processing {}...".format(video_file))
    cap = cv2.VideoCapture(video_file)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        print("Warning: Could not determine FPS for {}. Using 25 as default.".format(video_file))
        original_fps = 25.0  # fallback default

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_folder = os.path.join(args.asset_dir, "frames", "{}".format(video_name))
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0                     # total frame count in the video
    output_frame_index = 0              # frame index measured in desired fps
    next_extraction_time = 0.0          # next time (in seconds) to extract a frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break                       # end of video

        current_time = frame_count / original_fps

        if current_time >= next_extraction_time:
            image_filename = os.path.join(output_folder, "{}.png".format(output_frame_index))
            cv2.imwrite(image_filename, frame)
            print("Saved frame at {:.2f}s as {}".format(current_time, image_filename))
            output_frame_index += 1
            next_extraction_time += interval

        frame_count += 1

    cap.release()
    print("Finished processing {}. Frames saved in {}".format(video_file, output_folder))
