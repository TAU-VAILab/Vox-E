import cv2
import argparse
import imageio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="path to input video file")
parser.add_argument("-r", "--ref", help="path to reference video file")
parser.add_argument("-o", "--output", help="path to output file")

def concatenate_videos(video1_path, video2_path, output_path):
    # Open the first video file
    video1 = cv2.VideoCapture(video1_path)
    # Open the second video file
    video2 = cv2.VideoCapture(video2_path)
    
    # Get the video properties
    fps = int(video1.get(cv2.CAP_PROP_FPS))
    width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create the video writer object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #writer = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    # Loop over the frames in the videos
    rendered_frames = []
    while True:
        # Read a frame from each video
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        
        # Check if both frames were read successfully
        if ret1 and ret2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            # Concatenate the frames side by side
            concatenated_frame = cv2.hconcat([frame1, frame2])
            # Write the concatenated frame to the output video file
            #writer.write(concatenated_frame)
            rendered_frames.append(concatenated_frame)
        else:
            # Either the end of the first video or the second video has been reached
            break
            
    frames = np.stack(rendered_frames)
    imageio.mimwrite(
        output_path,
        frames,
        fps=fps,
    )
    
    # Release the video objects and writer object
    video1.release()
    video2.release()
    #writer.release()


if __name__ == "__main__":
    args = parser.parse_args()
    concatenate_videos(args.ref, args.input, args.output)