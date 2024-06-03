################################
# Name: Kahlawy Hussein
# Acadmic Number: 202301531
# Software Engineering Master - First Year (2023/2024)
# Supervised by Dr. Ashraf A. Shahin 
################################

import numpy as np
import os
import cv2
from xml.etree import ElementTree as et
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import logging
from mean_average_precision import MetricBuilder
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

# Constants
BATCH_SIZE = 8
RESIZE_TO = 512
NUM_EPOCHS = 50
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TRAIN_DIR = 'C:/kaggle/input/train'
VALID_DIR = 'C:/kaggle/input/val'
CLASSES = ['crack', 'damage', 'pothole', 'pothole_water', 'pothole_water_m']
NUM_CLASSES = len(CLASSES) + 1  # Add 1 for background class
OUT_DIR = 'C:/kaggle/outputs'
SAVE_PLOTS_EPOCH = 5
SAVE_MODEL_EPOCH = 5

# Albumentations transforms
def get_train_transform():
    return A.Compose([
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.MotionBlur(p=0.2, always_apply=False),
        A.MedianBlur(blur_limit=3, p=0.1, always_apply=False),
        A.Blur(blur_limit=3, p=0.1, always_apply=False),
        ToTensorV2(p=1.0, always_apply=True),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0, always_apply=True),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# Dataset class
class PotholeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))
        self.xml_paths = glob.glob(os.path.join(root_dir, "*.xml"))
    
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
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Logging
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

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

# Training and validation functions
def train(train_data_loader, model):
    print('Training')
    global train_itr
    
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        losses.backward()
        optimizer.step()
        
        train_itr += 1
        
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    return loss_value

def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        val_itr += 1
        
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    return loss_value

# Evaluation
def evaluate(valid_loader, model, metric_fn):
    prog_bar = tqdm(valid_loader, total=len(valid_loader))
    
    voc_map = []
    voc_all_map = []
    coco_map = []
    
    model.eval()
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images)
        
        metric_fn.add(outputs, targets)
        
        batch_voc_map = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1), mpolicy='soft')['mAP']
        batch_voc_all_map = metric_fn.value(iou_thresholds=0.5, mpolicy='soft')['mAP']
        batch_coco_map = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
        
        voc_map.append(batch_voc_map)
        voc_all_map.append(batch_voc_all_map)
        coco_map.append(batch_coco_map)
    
    mean_voc_map = np.mean(voc_map)
    mean_voc_all_map = np.mean(voc_all_map)
    mean_coco_map = np.mean(coco_map)
    
    print(f"VOC PASCAL mAP: {mean_voc_map:.4f}")
    print(f"VOC PASCAL mAP in all points: {mean_voc_all_map:.4f}")
    print(f"COCO mAP: {mean_coco_map:.4f}")

if __name__ == '__main__':
    # Call freeze_support() if you are using Windows and intend to freeze the program
    # multiprocessing.freeze_support()

    # Prepare datasets and data loaders
    train_dataset = PotholeDataset(TRAIN_DIR, transform=get_train_transform())
    valid_dataset = PotholeDataset(VALID_DIR, transform=get_valid_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}")
    

    # Visualize 10 sample images  training
    num_samples = 10
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(num_samples):
        sample_image, sample_target = train_dataset[i]
        sample_boxes = sample_target['boxes'].numpy()
        sample_labels = sample_target['labels'].numpy()
        
        ax = axes[i // 5, i % 5]
        ax.imshow(sample_image.permute(1, 2, 0).numpy())
        
        for box, label in zip(sample_boxes, sample_labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, CLASSES[label], fontsize=12, color='r', bbox=dict(facecolor='white', alpha=0.5))
        
        ax.axis('off')

    plt.suptitle("Sample of Image Detection")
    plt.tight_layout()
    plt.show()

    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Training loop
    train_itr = 1
    val_itr = 1
    best_valid_loss = float('inf')
    model_name = 'pothole_detection_model'

    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        
        start = time.time()
        train_loss = train(train_loader, model)
        valid_loss = validate(valid_loader, model)
        
        scheduler.step(valid_loss)
        last_lr = scheduler.optimizer.param_groups[0]['lr']
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{model_name}_best.pth"))
        
        end = time.time()
        print(f"Epoch #{epoch+1} train loss: {train_loss:.3f}")
        print(f"Epoch #{epoch+1} validation loss: {valid_loss:.3f}")
        print(f"Last learning rate: {last_lr:.6f}")
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")
        
        if (epoch + 1) % SAVE_MODEL_EPOCH == 0:
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{model_name}_{epoch+1}.pth"))
            print('SAVING MODEL COMPLETE...\n')

    # Load the best model
    best_model = create_model(num_classes=NUM_CLASSES)
    best_model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{model_name}_best.pth")))
    best_model.to(DEVICE)
    best_model.eval()

    # Visualize 10 sample images after training
    num_samples = 10
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(num_samples):
        sample_image, _ = train_dataset[i]
        sample_image_tensor = sample_image.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = best_model(sample_image_tensor)
        
        predicte
        