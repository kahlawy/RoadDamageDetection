################################
# Kahlawy Hussein
# Acadmic Code: 202301531
# Software Engineering Master - First Year (2023/2024)
################################

import numpy as np
import os
import cv2
from xml.etree import ElementTree as et
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from mean_average_precision import MetricBuilder
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")

# Constants
BATCH_SIZE = 8
RESIZE_TO = 224
NUM_EPOCHS = 2
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TRAIN_DIR = 'C:/kaggle/input/train'
VALID_DIR = 'C:/kaggle/input/val'
CLASSES = ['crack', 'damage', 'pothole', 'pothole_water', 'pothole_water_m']
NUM_CLASSES = len(CLASSES) + 1  # Add 1 for background class
OUT_DIR = 'C:/kaggle/outputs'
SAVE_PLOTS_EPOCH = 1
SAVE_MODEL_EPOCH = 1

# Albumentations transforms
def get_train_transform():
    return A.Compose([
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Dataset class
class PotholeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.xml_paths = sorted(glob.glob(os.path.join(root_dir, "*.xml")))
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (RESIZE_TO, RESIZE_TO))
        image_resized /= 255.0
        
        xml_path = self.xml_paths[idx]
        boxes, labels = self.extract_boxes_and_labels(xml_path, image.shape)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        
        if self.transform:
            sample = self.transform(image=image_resized, bboxes=target['boxes'], labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'])
        
        return image_resized, target
    
    def __len__(self):
        return len(self.image_paths)
    
    def extract_boxes_and_labels(self, xml_path, image_shape):
        boxes = []
        labels = []
        
        tree = et.parse(xml_path)
        root = tree.getroot()
        
        image_height, image_width, _ = image_shape
        
        for member in root.findall('object'):
            label = CLASSES.index(member.find('name').text)
            labels.append(label)
            
            xmin = int(member.find('bndbox').find('xmin').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            xmin_final = (xmin / image_width) * RESIZE_TO
            ymin_final = (ymin / image_height) * RESIZE_TO
            xmax_final = (xmax / image_width) * RESIZE_TO
            ymax_final = (ymax / image_height) * RESIZE_TO
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        return boxes, labels

# Model
def create_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Logging
logging.basicConfig(filename='C:/kaggle/outputs/train.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Visualization function
def visualize_image_with_boxes(image, boxes, labels):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, CLASSES[label], fontsize=12, color='r', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.axis('off')
    plt.show()

# Function to plot training and validation loss
def plot_loss(train_losses, valid_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, valid_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()



# Training and validation functions
def train(train_data_loader, model, optimizer, lr_scheduler):
    model.train()
    total_loss = 0
    for images, targets in tqdm(train_data_loader, desc="Training"):
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    lr_scheduler.step(total_loss / len(train_data_loader))
    return total_loss / len(train_data_loader)

def validate(valid_data_loader, model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(valid_data_loader, desc="Validation"):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    return total_loss / len(valid_data_loader)

# Evaluation
def evaluate(valid_loader, model, metric_fn):
    model.eval()
    for images, targets in tqdm(valid_loader, desc="Evaluating"):
        images = list(image.to(DEVICE) for image in images)
        outputs = model(images)
        for i, output in enumerate(outputs):
            gt_boxes = targets[i]['boxes'].cpu().numpy()
            gt_labels = targets[i]['labels'].cpu().numpy()
            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()
            metric_fn.add(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    return metric_fn.value(iou_thresholds=0.5)

# Load data
train_dataset = PotholeDataset(TRAIN_DIR, transform=get_train_transform())
valid_dataset = PotholeDataset(VALID_DIR, transform=get_valid_transform())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Create model, optimizer, and learning rate scheduler
model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)

train_losses = []
valid_losses = []

for epoch in range(NUM_EPOCHS):
    logging.info(f"Epoch {epoch+1} of {NUM_EPOCHS}")
    train_loss = train(train_loader, model, optimizer, lr_scheduler)
    logging.info(f"Training loss: {train_loss}")
    valid_loss = validate(valid_loader, model)
    logging.info(f"Validation loss: {valid_loss}")
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    if (epoch + 1) % SAVE_MODEL_EPOCH == 0:
        model_save_path = os.path.join(OUT_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Saved model at {model_save_path}")
    
    if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:
        for i in range(6):  # Display 6 samples instead of 5
            image, target = valid_dataset[i]
            visualize_image_with_boxes(
                image.permute(1, 2, 0).cpu().numpy(),
                target['boxes'].cpu().numpy(),
                target['labels'].cpu().numpy()
            )
            logging.info(f"Displayed visualized image at epoch {epoch+1}, sample {i}")

# Plot training and validation loss
plot_loss(train_losses, valid_losses)
logging.info("Displayed loss plot.")

logging.info("Training and validation completed.")

# Evaluation
metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=NUM_CLASSES)
eval_result = evaluate(valid_loader, model, metric_fn)
logging.info(f"Evaluation result: {eval_result}")