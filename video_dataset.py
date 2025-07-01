import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image  # Import PIL.Image to convert NumPy arrays to PIL images


class VideoDataset(Dataset):
    def __init__(self, csv_file, video_dir, num_frames=60, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with video information.
            video_dir (string): Directory with all the videos.
            num_frames (int): Number of frames to sample from each video.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize all frames to 224x224
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Tensor of video frames.
            torch.Tensor: Label for the video.
        """
        row = self.data.iloc[idx]
        video_file = os.path.join(self.video_dir, row['file'])  # 'file' column in CSV
        label = row['label']

        frames = self._load_video_frames(video_file)

        # Convert frames to PIL images and apply transformations
        frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
        frames = [self.transform(frame) for frame in frames]  # Apply transformations to each frame

        # Ensure all frames are stacked into a tensor (T, C, H, W)
        video_tensor = torch.stack(frames)  # Stack frames into a tensor (T, C, H, W)

        return video_tensor, torch.tensor(label, dtype=torch.long)

    def _load_video_frames(self, video_path):
        """
        Loads frames from the video file and samples frames if necessary.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List of frames (each a numpy array).
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from {video_path}")

        # If the video has more than num_frames, sample exactly num_frames evenly
        if len(frames) > self.num_frames:
            idxs = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in idxs]
        # If the video has fewer than num_frames, pad with the first frame
        elif len(frames) < self.num_frames:
            frames = frames + [frames[-1]] * (self.num_frames - len(frames))  # Padding with last frame

        return frames
