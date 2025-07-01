import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from video_dataset import VideoDataset
from model_resnettcn import ResNetTCN
import numpy as np

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths
    test_csv = "test_n.csv"
    video_dir = "E:/face_foren/face_only_video"
    model_path = "resnet_tcn_deepfake_1.pt"
    graph_folder = "graph"
    os.makedirs(graph_folder, exist_ok=True)

    # Dataset & DataLoader
    test_dataset = VideoDataset(test_csv, video_dir)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)

    # Load model
    model = ResNetTCN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluation
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_probs = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(test_loader):
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for class 1 (Fake)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Get filenames for current batch
            batch_filenames = test_dataset.data['file'][batch_idx * test_loader.batch_size : batch_idx * test_loader.batch_size + labels.size(0)].tolist()
            all_filenames.extend(batch_filenames)

    # Metrics
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)

    print("\n🧪 Test Results")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Loss: {avg_loss:.4f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

    # Save ROC Curve
    plot_roc_curve(all_labels, all_probs, graph_folder)

    # Save Confusion Matrix
    plot_confusion_matrix(cm, ["Real", "Fake"], graph_folder)

def plot_confusion_matrix(cm, class_names, folder):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "confusion_matrix.png"))
    plt.close()
    print(f"✅ Confusion Matrix saved to {os.path.join(folder, 'confusion_matrix.png')}")

def plot_roc_curve(y_true, y_probs, folder):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Deepfake Detection')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "roc_curve.png"))
    plt.close()
    print(f"✅ ROC Curve saved to {os.path.join(folder, 'roc_curve.png')}")

if __name__ == "__main__":
    test_model()
