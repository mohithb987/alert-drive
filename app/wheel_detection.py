import os
import glob
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def get_relative_file_paths(directory, root_folder):
    root_folder_abs_path = os.path.abspath(root_folder)
    all_files = glob.glob(os.path.join(directory, '*/*/*'), recursive=True)
    relative_paths = [
        os.path.relpath(file, root_folder_abs_path) for file in all_files if os.path.isfile(file) and file.endswith('.jpg')
    ]
    shotgun_files = glob.glob(os.path.join(directory, '*/shotgun/*'), recursive=True)
    shotgun_relative_paths = [
        os.path.relpath(file, root_folder_abs_path) for file in shotgun_files if os.path.isfile(file) and file.endswith('.jpg')
    ]
    driver_files = glob.glob(os.path.join(directory, '*/driver/*'), recursive=True)
    driver_relative_paths = [
        os.path.relpath(file, root_folder_abs_path) for file in driver_files if os.path.isfile(file) and file.endswith('.jpg')
    ]
    shotgun_data = {path: 0 for path in shotgun_relative_paths}
    driver_data = {path: 1 for path in driver_relative_paths}
    all_data = {**driver_data, **shotgun_data}

    print('All files:', len(relative_paths))
    print('Class driver:', len(driver_relative_paths))
    print('Class shotgun:', len(shotgun_relative_paths))
    print('Driver Data:', driver_data)
    print('Shotgun Data:', shotgun_data)
    print('All Data:', all_data)
    
    return all_data

def load_and_augment_images(all_data, save_dir='../input/data/steering_wheel/augmented_images'):
    import cv2
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    
    root_dir = '../'
    augmented_images = []
    augmented_labels = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, (file_path, label) in enumerate(all_data.items()):
        img = Image.open(os.path.join(root_dir, file_path))
        if idx % 2 == 1:
            img = transforms.functional.hflip(img)
        transformed_img = transform(img)
        numpy_img = (transformed_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bgr_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f'_{idx}.jpg'), bgr_img)
        augmented_images.append(transformed_img)
        augmented_labels.append(label)
    
    print('Len augmented images:', len(augmented_images))
    print('Len augmented augmented_labels:', len(augmented_labels))

    return torch.stack(augmented_images), np.array(augmented_labels)

import torch
import torch.nn as nn
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten_dim = self._get_flatten_dim(input_shape)
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def _get_flatten_dim(self, shape):
        x = torch.randn(1, *shape)
        x = self.features(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        probabilities = self.softmax(logits)
        return probabilities




def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, patience=5):
    best_val_loss = np.inf
    no_improvement_count = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_total += labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= patience:
            print(f'Early stopping after epoch {epoch+1}')
            break


def main():
    directory = '../input/data/images/'  
    root_folder = '../'
    
    all_data = get_relative_file_paths(directory, root_folder)
    X_augmented, y_augmented = load_and_augment_images(all_data)

    random_state = 42
    X_train, X_val, y_train, y_val = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=random_state)

    X_train, X_val = torch.tensor(X_train), torch.tensor(X_val)
    y_train, y_val = torch.tensor(y_train), torch.tensor(y_val)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    input_shape = (3, 256, 256)
    num_classes = 2
    model = SimpleCNN(input_shape, num_classes)
    print(model)

    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.Tensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print('Len Train Loader: ', len(train_loader))
    print('Len Val Loader: ', len(val_loader))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print('Starting to Train Model...')
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, patience=5)
    print('------ Finished Training the Model ------')
    model_path = '../model/trained_model/simple_cnn_model4.pth'
    torch.save(model, model_path)

if __name__ == "__main__":
    main()