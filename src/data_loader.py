import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
import random
import pandas as pd

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, covid_dir=None, additional_dir=None, nih_dir=None, bachrr_dir=None, split='train', transform=None):
        self.root_dir = root_dir
        self.covid_dir = covid_dir
        self.additional_dir = additional_dir
        self.nih_dir = nih_dir
        self.bachrr_dir = bachrr_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # --- 1. Load Pneumonia Dataset (Normal vs Pneumonia) ---
        # Structure: root_dir/train/NORMAL, root_dir/train/PNEUMONIA
        self.split_dir = os.path.join(root_dir, split)
        
        # Load Normal images (Label 0)
        normal_path = os.path.join(self.split_dir, 'NORMAL', '*.jpeg')
        normal_files = glob(normal_path)
        self.images.extend(normal_files)
        self.labels.extend([0] * len(normal_files))
        
        # Load Pneumonia images (Label 1)
        pneu_path = os.path.join(self.split_dir, 'PNEUMONIA', '*.jpeg')
        pneu_files = glob(pneu_path)
        self.images.extend(pneu_files)
        self.labels.extend([1] * len(pneu_files))
        
        # --- 2. Load COVID-19 Dataset (Label 2) AND Extra Normal/Pneumonia ---
        # Structure: covid_dir/COVID-19_Radiography_Dataset/COVID/images/*.png
        if self.covid_dir:
            # Helper to load and split files from a subdirectory
            def load_and_split(subdir, label):
                files = glob(os.path.join(self.covid_dir, '**', subdir, 'images', '*.png'), recursive=True)
                if not files:
                    files = glob(os.path.join(self.covid_dir, subdir, '*.png'))
                
                files.sort()
                random.seed(42)
                random.shuffle(files)
                
                n_total = len(files)
                n_train = int(n_total * 0.8)
                n_val = int(n_total * 0.1)
                
                if split == 'train':
                    return files[:n_train]
                elif split == 'val':
                    return files[n_train:n_train+n_val]
                else: # test
                    return files[n_train+n_val:]

            # Load COVID (Label 2)
            covid_files = load_and_split('COVID', 2)
            self.images.extend(covid_files)
            self.labels.extend([2] * len(covid_files))
            
            # Load Normal (Label 0) - Huge boost!
            normal_extra = load_and_split('Normal', 0)
            self.images.extend(normal_extra)
            self.labels.extend([0] * len(normal_extra))
            
            # Load Viral Pneumonia (Label 1) - Huge boost!
            pneu_extra = load_and_split('Viral Pneumonia', 1)
            self.images.extend(pneu_extra)
            self.labels.extend([1] * len(pneu_extra))

        # --- 3. Load Additional Dataset (Prashant268) ---
        if self.additional_dir:
            # Structure: additional_dir/train/COVID19, additional_dir/train/NORMAL, etc.
            # Map split to folder name
            target_folder = 'train' if split == 'train' else 'test'
            
            # If split is 'val', we can use 'test' or a subset of 'train'. 
            # For simplicity, let's use 'test' for both val and test to avoid overlap with train
            if split == 'val':
                target_folder = 'test'
                
            base_path = os.path.join(self.additional_dir, target_folder)
            
            # Normal (Label 0)
            self.images.extend(glob(os.path.join(base_path, 'NORMAL', '*.jpg')))
            self.labels.extend([0] * len(glob(os.path.join(base_path, 'NORMAL', '*.jpg'))))
            
            # Pneumonia (Label 1)
            self.images.extend(glob(os.path.join(base_path, 'PNEUMONIA', '*.jpg')))
            self.labels.extend([1] * len(glob(os.path.join(base_path, 'PNEUMONIA', '*.jpg'))))
            
            # COVID19 (Label 2)
            self.images.extend(glob(os.path.join(base_path, 'COVID19', '*.jpg')))
            self.labels.extend([2] * len(glob(os.path.join(base_path, 'COVID19', '*.jpg'))))
            
        # --- 4. Load NIH Dataset (Adults) ---
        if self.nih_dir:
            # Structure: nih_dir/sample/images/*.png
            # Labels: nih_dir/sample_labels.csv
            csv_path = os.path.join(self.nih_dir, 'sample_labels.csv')
            images_dir = os.path.join(self.nih_dir, 'sample', 'images')
            
            if os.path.exists(csv_path) and os.path.exists(images_dir):
                df = pd.read_csv(csv_path)
                
                # Filter for Pneumonia and Normal (No Finding)
                # Note: NIH labels are multi-label separated by '|'
                
                # Get Pneumonia cases
                pneumonia_df = df[df['Finding Labels'].str.contains('Pneumonia')]
                
                # Get Normal cases (No Finding)
                normal_df = df[df['Finding Labels'] == 'No Finding']
                
                # Split logic for NIH
                # We'll use a simple hash-based split or random split to ensure consistency
                # Or just use the same random seed
                
                def get_split_files(dataframe):
                    files = dataframe['Image Index'].tolist()
                    files = [os.path.join(images_dir, f) for f in files]
                    # Filter existing files
                    files = [f for f in files if os.path.exists(f)]
                    
                    files.sort()
                    random.seed(42)
                    random.shuffle(files)
                    
                    n_total = len(files)
                    n_train = int(n_total * 0.8)
                    n_val = int(n_total * 0.1)
                    
                    if split == 'train':
                        return files[:n_train]
                    elif split == 'val':
                        return files[n_train:n_train+n_val]
                    else: # test
                        return files[n_train+n_val:]

                # Add Pneumonia (Label 1)
                pneu_files = get_split_files(pneumonia_df)
                self.images.extend(pneu_files)
                self.labels.extend([1] * len(pneu_files))
                
                # Add Normal (Label 0)
                # Since NIH has MANY normals, we might want to limit them to balance classes if needed
                # But for now, let's add them all to help with "Adult Normal" recognition
                normal_files = get_split_files(normal_df)
                self.images.extend(normal_files)
                self.labels.extend([0] * len(normal_files))

        # --- 5. Load Bachrr COVID-19 Dataset ---
        if self.bachrr_dir:
            # Structure: bachrr_dir/images/*.jpeg
            # Metadata: bachrr_dir/metadata.csv
            csv_path = os.path.join(self.bachrr_dir, 'metadata.csv')
            images_dir = os.path.join(self.bachrr_dir, 'images')
            
            if os.path.exists(csv_path) and os.path.exists(images_dir):
                df = pd.read_csv(csv_path)
                
                # Filter for COVID-19
                # The 'finding' column contains 'COVID-19'
                covid_df = df[df['finding'].str.contains('COVID-19', na=False)]
                
                def get_split_files_bachrr(dataframe):
                    files = dataframe['filename'].tolist()
                    files = [os.path.join(images_dir, f) for f in files]
                    # Filter existing files
                    files = [f for f in files if os.path.exists(f)]
                    
                    files.sort()
                    random.seed(42)
                    random.shuffle(files)
                    
                    n_total = len(files)
                    n_train = int(n_total * 0.8)
                    n_val = int(n_total * 0.1)
                    
                    if split == 'train':
                        return files[:n_train]
                    elif split == 'val':
                        return files[n_train:n_train+n_val]
                    else: # test
                        return files[n_train+n_val:]
                
                # Add COVID-19 (Label 2)
                covid_files = get_split_files_bachrr(covid_df)
                self.images.extend(covid_files)
                self.labels.extend([2] * len(covid_files))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

def get_transforms(split):
    if split == 'train':
        return A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            # More aggressive augmentation to prevent overfitting to source artifacts
            A.Rotate(limit=30, p=0.7), # Increased limit and probability
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6), # Increased limits
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.15, 0.15), rotate=0, p=0.4), # Increased scale range
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

def get_dataloaders(data_dir, covid_dir=None, additional_dir=None, nih_dir=None, bachrr_dir=None, batch_size=32, num_workers=4):
    datasets = {
        x: ChestXRayDataset(data_dir, covid_dir=covid_dir, additional_dir=additional_dir, nih_dir=nih_dir, bachrr_dir=bachrr_dir, split=x, transform=get_transforms(x))
        for x in ['train', 'test', 'val']
    }
    
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=num_workers)
        for x in ['train', 'test', 'val']
    }
    
    return dataloaders, datasets
