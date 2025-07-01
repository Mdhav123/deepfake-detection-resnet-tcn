import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def crop_and_save_face_video(input_frame_root, output_video_root, model_path):
    model = YOLO(model_path)

    os.makedirs(output_video_root, exist_ok=True)
    video_folders = sorted([d for d in os.listdir(input_frame_root) if os.path.isdir(os.path.join(input_frame_root, d))])

    for video_folder in tqdm(video_folders, desc=f"Processing {input_frame_root.split('/')[-1]}"):
        video_frame_path = os.path.join(input_frame_root, video_folder)
        frame_files = sorted([f for f in os.listdir(video_frame_path) if f.endswith('.jpg')])

        face_frames = []

        for frame_file in frame_files:
            frame_path = os.path.join(video_frame_path, frame_file)
            image = cv2.imread(frame_path)

            results = model(image)[0]
            boxes = results.boxes

            if boxes is not None and len(boxes) > 0:
                # Use the largest detected face
                biggest_box = max(boxes.xyxy, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                x1, y1, x2, y2 = map(int, biggest_box)
                face = image[y1:y2, x1:x2]
                face_resized = cv2.resize(face, (112, 112))
                face_frames.append(face_resized)

        if face_frames:
            output_path = os.path.join(output_video_root, f"{video_folder}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 25  # You can tweak this
            out = cv2.VideoWriter(output_path, fourcc, fps, (112, 112))

            for face_frame in face_frames:
                out.write(face_frame)
            out.release()

# --- Config ---
yolo_model_path = "E:/dataset-FR/yolov8n-face-lindevs.pt"



# Process fake videos
crop_and_save_face_video(
    input_frame_root="E:/face_foren/frames/fake",
    output_video_root="E:/face_foren/face_only_video/fake",
    model_path=yolo_model_path
)
