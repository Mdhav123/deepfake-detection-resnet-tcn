import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from video_dataset import VideoDataset
from model_resnettcn import ResNetTCN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths
    train_csv = "train_n.csv"
    val_csv = "val_n.csv"
    video_dir = "E:/face_foren/face_only_video"

    # Dataset and DataLoader
    train_dataset = VideoDataset(train_csv, video_dir)
    val_dataset = VideoDataset(val_csv, video_dir)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    # Model
    model = ResNetTCN()
    model.to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Iterate through training data
        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Print metrics for the epoch
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_accuracy:.2f}%\n")

    # Save the trained model
    torch.save(model.state_dict(), "resnet_tcn_deepfake_1.pt")
    print("Model saved as resnet_tcn_deepfake_1.pt")


if __name__ == "__main__":
    main()
