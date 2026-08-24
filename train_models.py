import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(f"PyTorch Version: {torch.__version__}", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}", flush=True)

# 1. Load 32x32 dataset
data = np.load("Digit_Dataset_32x32/dataset_32x32.npz")
X_train_np, y_train_np = data["X_train"], data["y_train"]
X_val_np, y_val_np     = data["X_val"], data["y_val"]
X_test_np, y_test_np   = data["X_test"], data["y_test"]

print(f"Train samples: {X_train_np.shape}, Val samples: {X_val_np.shape}, Test samples: {X_test_np.shape}", flush=True)

# Convert to PyTorch Tensors (add channel dimension: (N, 1, 32, 32))
X_train = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train_np, dtype=torch.long)

X_val = torch.tensor(X_val_np, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val_np, dtype=torch.long)

X_test = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)


# 2. Model 1: LeNet-5 Architecture for 32x32 Input
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)  # 32x32 -> 28x28
        self.pool1 = nn.MaxPool2d(2, 2)                                     # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0) # 14x14 -> 10x10
        self.pool2 = nn.MaxPool2d(2, 2)                                     # 10x10 -> 5x5
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# 3. Model 2: MiniDigitCNN (Modern lightweight CNN with BatchNorm & Dropout)
class MiniDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniDigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) # 32x32 -> 16x16
        self.drop1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # 16x16 -> 8x8
        self.drop2 = nn.Dropout2d(0.3)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop2(self.pool2(x))

        x = x.view(x.size(0), -1)
        x = self.drop3(F.relu(self.bn_fc(self.fc1(x))))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, epochs=15, lr=0.001, model_name="model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = None

    print(f"\n--- Training {model_name} for {epochs} Epochs ---", flush=True)
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation Phase
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss_sum += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%", flush=True)

    model.load_state_dict(best_weights)
    print(f"[+] Best Validation Accuracy for {model_name}: {best_val_acc*100:.2f}%\n", flush=True)
    return model, history


# Train both models
lenet = LeNet5().to(device)
lenet, lenet_hist = train_model(lenet, train_loader, val_loader, epochs=15, lr=0.001, model_name="LeNet-5")

minidnn = MiniDigitCNN().to(device)
minidnn, minidnn_hist = train_model(minidnn, train_loader, val_loader, epochs=15, lr=0.001, model_name="MiniDigitCNN")

# Save the models
os.makedirs("models", exist_ok=True)
torch.save(minidnn.state_dict(), "models/best_digit_cnn.pt")
torch.save(lenet.state_dict(), "models/lenet5.pt")
print("[+] Saved model weights to models/best_digit_cnn.pt and models/lenet5.pt", flush=True)


# Evaluate on Untouched Test Set
def evaluate_test(model, test_loader, model_name="Model"):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    print(f"=======================================================", flush=True)
    print(f"  EVALUATION ON UNTOUCHED TEST SET: {model_name}", flush=True)
    print(f"  Test Accuracy: {acc*100:.2f}%", flush=True)
    print(f"=======================================================", flush=True)
    print(classification_report(all_targets, all_preds, digits=4), flush=True)
    return acc, all_preds, all_targets


acc1, preds1, targets1 = evaluate_test(lenet, test_loader, "LeNet-5")
acc2, preds2, targets2 = evaluate_test(minidnn, test_loader, "MiniDigitCNN")


# Plot and Save Training Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(lenet_hist["train_loss"], label="LeNet-5 Train", linestyle="--")
plt.plot(lenet_hist["val_loss"], label="LeNet-5 Val", linestyle="-")
plt.plot(minidnn_hist["train_loss"], label="MiniCNN Train", linestyle="--")
plt.plot(minidnn_hist["val_loss"], label="MiniCNN Val", linestyle="-")
plt.title("Loss Curves (Training vs Validation)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(lenet_hist["train_acc"], label="LeNet-5 Train", linestyle="--")
plt.plot(lenet_hist["val_acc"], label="LeNet-5 Val", linestyle="-")
plt.plot(minidnn_hist["train_acc"], label="MiniCNN Train", linestyle="--")
plt.plot(minidnn_hist["val_acc"], label="MiniCNN Val", linestyle="-")
plt.title("Accuracy Curves (Training vs Validation)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/training_curves.png", dpi=150)
plt.close()
print("[+] Saved training curves to models/training_curves.png", flush=True)


# Plot and Save Confusion Matrix for Best Model (MiniDigitCNN)
cm = confusion_matrix(targets2, preds2)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])
plt.title(f"Confusion Matrix on Untouched Test Set (Accuracy: {acc2*100:.2f}%)")
plt.xlabel("Predicted Digit Label")
plt.ylabel("True Digit Label")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.close()
print("[+] Saved confusion matrix to models/confusion_matrix.png", flush=True)


# Plot and Save Test Prediction Samples
plt.figure(figsize=(12, 6))
indices = np.random.RandomState(42).choice(len(X_test_np), size=min(18, len(X_test_np)), replace=False)
for idx, i in enumerate(indices):
    plt.subplot(3, 6, idx + 1)
    plt.imshow(X_test_np[i], cmap="gray")
    true_l = y_test_np[i]
    pred_l = preds2[i]
    color = "green" if true_l == pred_l else "red"
    plt.title(f"True: {true_l} | Pred: {pred_l}", color=color, fontsize=10)
    plt.axis("off")

plt.suptitle("Sample Predictions on Untouched Test Set (MiniDigitCNN)", fontsize=13)
plt.tight_layout()
plt.savefig("models/test_predictions_sample.png", dpi=150)
plt.close()
print("[+] Saved test predictions preview to models/test_predictions_sample.png", flush=True)
