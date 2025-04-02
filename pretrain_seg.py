import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import models
from dataset import get_dataset
from models import *
from loss_functions import *
from miou import *
from torch.cuda.amp import GradScaler, autocast


scaler = GradScaler()

# Hyperparameters
batch_size = 64
imsize = (1024, 512)
num_iterations = 10000
save_interval = 100  # Save every 100 minibatch steps
best_miou = 0.0

# Load dataset
traindataset, testdataset = get_dataset("G_pretrain", batch=batch_size, imsize=imsize, workers=0)
train_loader = iter(traindataset)  # Convert DataLoader to an iterator

# Model, optimizer, loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = drn26().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

# Training loop with minibatch iterations
iteration = 0

while iteration < num_iterations:
    net.train()

    try:
        imgs, labels = next(train_loader)  # Fetch next minibatch
    except StopIteration:
        # If dataset is exhausted, restart from the beginning
        train_loader = iter(traindataset)
        imgs, labels = next(train_loader)

    imgs, labels = imgs.to(device), labels.to(device).long()

    optimizer.zero_grad()

    with autocast():
        outputs = net(imgs)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Calculate training mIoU
    with torch.no_grad():
        confusion_matrix = np.zeros((19,) * 2)
        pred = outputs.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        gt = labels.data.cpu().numpy()
        confusion_matrix += MIOU(gt, pred)

        score = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
        train_miou = np.nanmean(score) * 100.

    print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.4f}, mIoU: {train_miou:.4f}")
    print(f"Scaler scale: {scaler.get_scale()}")
    print(", ".join([f"{val * 100:.2f}" for val in score]))


    # Evaluation & model saving at intervals
    if (iteration + 1) % save_interval == 0:
        net.eval()
        total_miou = 0.0
        confusion_matrix = np.zeros((19,) * 2)

        with torch.no_grad():
            for imgs, labels in testdataset:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = net(imgs)
                pred = outputs.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                gt = labels.data.cpu().numpy()
                confusion_matrix += MIOU(gt, pred)

            score = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
            miou = np.nanmean(score) * 100.

        print(f"Iteration {iteration + 1}/{num_iterations}, Validation mIoU: {miou:.4f}")

        # Save best model
        if miou > best_miou:
            best_miou = miou
            save_path = f"./pretrained/drn26_iter{iteration + 1}.pth"
            torch.save(net.state_dict(), save_path)
            print(f"Saved model to {save_path}")

        net.train()

    iteration += 1  # Increment after each minibatch step



