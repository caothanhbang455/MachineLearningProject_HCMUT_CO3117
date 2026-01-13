import os
import torch
import wandb
import torch.nn as nn
from datetime import datetime
from models.mlp import MLPClassifier
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter

def test_feature(model, X_test_t, y_test_t, num_classes=10):
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        y_pred = preds.argmax(dim=1)
        acc = (y_pred == y_test_t).float().mean().item()

    print(f"\n Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_t.cpu(), y_pred.cpu(), digits=4))
    return acc



def train_feature(
    X_train_t,
    y_train_t,
    X_val_t=None,          # optional validation
    y_val_t=None,
    input_dim=128,
    num_classes=10,
    lr=1e-3,
    epochs=100,
    patience=3,
    use_wandb=False
):
    model = MLPClassifier(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/mlp_feature_{timestamp}.pt"

    # Logging
    writer = SummaryWriter(log_dir=f"runs/mlp_feature_{timestamp}")

    if use_wandb:
        wandb.init(project="audio-feature-mlp", config={
            "lr": lr, "epochs": epochs, "architecture": "MLP"
        })
        wandb.watch(model)

    # Early stopping state
    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()

        # Default val metrics = training metrics (no val provided)
        val_loss = loss.item()
        val_acc = (preds.argmax(dim=1) == y_train_t).float().mean().item()

        # If val set available, override
        if X_val_t is not None and y_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()
                val_acc = (val_preds.argmax(dim=1) == y_val_t).float().mean().item()

        print(f"[{epoch+1}/{epochs}] "
              f"Train Loss: {loss.item():.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # TensorBoard
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # W&B
        if use_wandb:
            wandb.log({
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break

    writer.close()
    if use_wandb:
        wandb.finish()

    # ---- Load best model back ----
    model.load_state_dict(torch.load(save_path))
    print(f"\nBest model loaded from: {save_path}")

    return model, save_path

