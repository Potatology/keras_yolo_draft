import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torchvision.transforms.functional as TF


script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_IMAGES = os.path.join(script_dir, "../Data_labeling_project/labeled_dataset/poc_dataset_v2/images")
PATH_ANNOTATIONS = os.path.join(script_dir,"../Data_labeling_project/labeled_dataset/poc_dataset_v2/Annotations")

# Class mapping (Your custom classes)
class_mapping = {
    0: "act_num", 1: "amt", 2: "amt_wrt", 3: "bank_logo", 4: "check_num",
    5: "date", 6: "memo", 7: "nme_adrs", 8: "pay_order_of", 9: "rout", 10: "signature"
}

# Path to images and annotations (Your custom paths)
path_images = PATH_IMAGES
path_annot = PATH_ANNOTATIONS
SCORE_THRESHOLD = 0.6

class CheckDataset(torch.utils.data.Dataset):
    def __init__(self, root_images, root_annot, transforms=None):
        self.root_images = root_images
        self.root_annot = root_annot
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(root_images)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_images, self.imgs[idx])
        annot_path = os.path.join(self.root_annot, self.imgs[idx].replace('.jpg', '.xml'))
        img = Image.open(img_path).convert("RGB")
        target = {}
        target['boxes'] = []
        target['labels'] = []
        target['image_id'] = torch.tensor([idx])

        tree = ET.parse(annot_path)
        root = tree.getroot()
        for member in root.findall('object'):
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            target['boxes'].append([xmin, ymin, xmax, ymax])
            target['labels'].append(list(class_mapping.values()).index(member[0].text))

        # Apply image and bbox augmentation
        if self.transforms is not None:
            img = self.transforms(img)

        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        return img, target

# Define transforms
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

def freeze_model_layers(model, freeze_backbone=True, freeze_fpn=False, freeze_rpn=False, freeze_roi_head=False):
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    if freeze_fpn and hasattr(model, "backbone") and hasattr(model.backbone, "fpn"):
        for param in model.backbone.fpn.parameters():
            param.requires_grad = False
    if freeze_rpn:
        for param in model.rpn.parameters():
            param.requires_grad = False
    if freeze_roi_head:
        for param in model.roi_heads.parameters():
            param.requires_grad = False

# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Example: freeze backbone, train rest
freeze_model_layers(model, freeze_backbone=True, freeze_fpn=False, freeze_rpn=False, freeze_roi_head=False)

num_classes = len(class_mapping) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Create the data loader (batch_size=1)
dataset = CheckDataset(path_images, path_annot, transforms=transforms) # Apply transforms here
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda batch: tuple(zip(*batch)))

# Training loop (fixed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Reduced learning rate
num_epochs = 8

for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    loss_str = ', '.join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
    print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {losses.item():.4f}, {loss_str}")



torch.save(model.state_dict(), 'trained_model.pth')
print("Model saved as trained_model.pth")


# Prediction loop (fixed)
def predict_on_training_data():
    model.eval()
    for images, targets in data_loader:
        images_cpu = [image.cpu() for image in images]
        images_pil = [F.to_pil_image(image) for image in images_cpu]

        images = list(image.to(device) for image in images)
        with torch.no_grad():
            predictions = model(images)
        
        prediction = predictions[0] # Access the first (and only) prediction in the batch
        boxes = prediction['boxes'].cpu().numpy().tolist()
        labels = prediction['labels'].cpu().numpy().tolist()
        scores = prediction['scores'].cpu().numpy().tolist()
        
        print("Image Predictions:")
        for box, label, score in zip(boxes, labels, scores):
            if score > SCORE_THRESHOLD:
                print(f"  Label: {class_mapping[label]}, Box: {box}, Score: {score}")

        # Visualization part
        image_pil = images_pil[0] # get the PIL Image
        image_cv = np.array(image_pil)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        for box, label, score in zip(boxes, labels, scores):
            if score > SCORE_THRESHOLD:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label_text = f"{class_mapping[label]}: {score:.2f}"
                cv2.putText(image_cv, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Predictions", image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

# New class for prediction on single images
class CheckPredictor:
    def __init__(self, model_path, class_mapping, device='cpu'):
        self.device = torch.device(device)
        self.class_mapping = class_mapping
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # Example: freeze backbone, train rest
        num_classes = len(class_mapping) + 1
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def crop_bbox(self, image, box):
        x_min, y_min, x_max, y_max = map(int, box)
        return image.crop((x_min, y_min, x_max, y_max))

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        prediction = predictions[0]
        boxes = prediction['boxes'].cpu().numpy().tolist()
        labels = prediction['labels'].cpu().numpy().tolist()
        scores = prediction['scores'].cpu().numpy().tolist()

        print("Image Predictions:")
        for box, label, score in zip(boxes, labels, scores):
            if score > SCORE_THRESHOLD:
                print(f"  Label: {self.class_mapping[label]}, Box: {box}, Score: {score}")

        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        for box, label, score in zip(boxes, labels, scores):
            if score > SCORE_THRESHOLD:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label_text = f"{self.class_mapping[label]}: {score:.2f}"
                cv2.putText(image_cv, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        for box, label, score in zip(boxes, labels, scores):
            if score > SCORE_THRESHOLD and self.class_mapping[label] == "amt":
                cropped = self.crop_bbox(image, box)
                cropped.save("cropped_amt.png")
                break

        cv2.imshow("Predictions", image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



predictor = CheckPredictor('trained_model.pth', class_mapping, device='cuda' if torch.cuda.is_available() else 'cpu')
predictor.predict(os.path.join(script_dir,'../Data_labeling_project/manual_test_dataset/test_check_2.png'))
#predict_on_training_data()