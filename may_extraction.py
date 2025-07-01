import cv2
import os
from tqdm import tqdm

def extract_frames_real(video_folder, save_root, target_fps=20):
    """
    Extracts frames from real videos at target FPS and saves them in:
    E:/face_foren/frames/real/{video_name}/
    """
    os.makedirs(save_root, exist_ok=True)
    videos = [v for v in os.listdir(video_folder) if v.endswith(('.mp4', '.avi', '.mov'))]

    for video in tqdm(videos, desc="Extracting from real videos"):
        video_name = os.path.splitext(video)[0]
        save_path = os.path.join(save_root, video_name)
        os.makedirs(save_path, exist_ok=True)

        video_path = os.path.join(video_folder, video)
        vidcap = cv2.VideoCapture(video_path)

        original_fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps // target_fps) if original_fps > target_fps else 1

        frame_count = 0
        saved_count = 0

        while True:
            success, image = vidcap.read()
            if not success:
                break
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(save_path, f"{video_name}_frame{saved_count:03d}.jpg")
                cv2.imwrite(frame_filename, image)
                saved_count += 1
            frame_count += 1

        print(f"{video_name}: Extracted {saved_count} frames (target FPS: {target_fps})")
# Set paths
real_video_path = "E:/face_foren/all_real"              # Folder with input videos
real_save_root = "E:/face_foren/frames/real"            # Folder where extracted frames will be saved

# Call the function
extract_frames_real(
    video_folder=real_video_path,
    save_root=real_save_root,
    target_fps=10
)
