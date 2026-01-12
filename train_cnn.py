import os
import glob
import torch
import torch.nn as nn
from PIL import Image
from models.cnn import BasicCNN
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights

label2number = {
    'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
    'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9
}


class ImageDataset(Dataset):
    def __init__(self, img_files, labels):
        self.img_files = img_files
        self.labels = labels
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert("RGB")
        y = self.labels[idx]
        img = self.transform(img)
        return img, y

def train_image(path_data, num_classes, epochs=20, batch_size=32, test_ratio=0.15, val_ratio=0.15):

    paths = []
    labels = []

    for genre in os.listdir(path_data):
        genre_dir = os.path.join(path_data, genre)
        files = glob.glob(f"{genre_dir}/*")
        paths.extend(files)
        labels.extend([label2number[genre]] * len(files))

    # ==== Stratify Train / Temp ====
    paths_train, paths_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=(test_ratio + val_ratio),
        stratify=labels, random_state=42
    )

    val_ratio_adjusted = val_ratio / (test_ratio + val_ratio)
    paths_val, paths_test, y_val, y_test = train_test_split(
        paths_temp, y_temp, test_size=1 - val_ratio_adjusted,
        stratify=y_temp, random_state=42
    )

    train_size = len(paths_train)
    val_size   = len(paths_val)
    test_size  = len(paths_test)

    train_ds = ImageDataset(paths_train, y_train)
    val_ds   = ImageDataset(paths_val, y_val)
    test_ds  = ImageDataset(paths_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = BasicCNN(num_classes).to(device)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    for epoch in range(epochs):
        # ---------------- TRAIN ----------------
        total_loss = 0
        total_correct = 0
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (preds.argmax(dim=1) == y).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc  = total_correct / train_size

        # ---------------- VALIDATION ----------------
        val_loss, val_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
                val_correct += (preds.argmax(dim=1) == y).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_size

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # ---------------- TEST ----------------
    test_correct = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            test_loss += loss.item()
            test_correct += (preds.argmax(dim=1) == y).sum().item()

    test_loss /= len(test_loader)
    test_acc = test_correct / test_size

    # print("\nFINAL TEST RESULTS")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
