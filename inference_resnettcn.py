import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from model_resnettcn import ResNetTCN  # Your custom model class
import tempfile
import os
from PIL import Image

# ==== CONFIG ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_path = "E:/dataset-FR/real_videos/00052.mp4"
yolo_model_path = "E:/dataset-FR/yolov8n-face-lindevs.pt"
resnet_tcn_weights = "resnet_tcn_deepfake_1.pt"


yolo_model = YOLO(yolo_model_path)

# ==== Step 2: Detect Faces & Create Face-only Video ====
def create_face_only_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = 224, 224  # Resize target

    temp_video_path = os.path.join(tempfile.gettempdir(), "face_only_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (W, H))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(frame, verbose=False)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face_resized = cv2.resize(face, (W, H))
                out.write(face_resized)
                break  # One face per frame

    cap.release()
    out.release()
    return temp_video_path

# ==== Step 3: Load and Process Video ====
def load_video_tensor(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame indices to sample
    frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    idx = 0
    ret = True
    frame_id = 0

    while ret and len(frames) < num_frames:
        ret, frame = cap.read()
        if frame_id in frame_idxs:
            if frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame)
            frames.append(frame_tensor)
        frame_id += 1

    cap.release()

    # Pad if needed
    while len(frames) < num_frames:
        frames.append(torch.zeros_like(frames[0]))

    video_tensor = torch.stack(frames)  # Shape: (T, C, H, W)
    return video_tensor.unsqueeze(0)  # Add batch dim: (1, T, C, H, W)

# ==== Step 4: Inference ====
def predict(video_tensor):
    model = ResNetTCN().to(device)
    model.load_state_dict(torch.load(resnet_tcn_weights, map_location=device))
    model.eval()

    with torch.no_grad():
        video_tensor = video_tensor.to(device)
        output = model(video_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    return "Real" if pred == 0 else "Fake", confidence

# ==== Run Pipeline ====
if __name__ == "__main__":
    print("Processing video:", video_path)
    face_video = create_face_only_video(video_path)
    video_tensor = load_video_tensor(face_video)
    label, conf = predict(video_tensor)
    print(f"Prediction: {label} ({conf*100:.2f}% confidence)")
