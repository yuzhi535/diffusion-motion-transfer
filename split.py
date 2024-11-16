import cv2 as cv
import os
import click


# Function to split video into frames
@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path(exists=False))
def split_video_to_frames(video_path, output_folder):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the video file exists
    if not os.path.isfile(video_path):
        raise (f"Error: Video file '{video_path}' does not exist.")

    # delete old files in output folder
    for filename in os.listdir(output_folder):
        if filename.endswith(".png"):
            os.remove(os.path.join(output_folder, filename))
            print(f"Deleted {filename}")

    # Capture video
    cap = cv.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv.imwrite(frame_filename, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()
    print('split video to frames done')


if __name__ == "__main__":
    split_video_to_frames()
