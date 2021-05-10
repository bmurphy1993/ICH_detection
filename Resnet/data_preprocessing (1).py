#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing
# RSNA Intracranial Hemorrhage Detection
# * Create dataset for pytorch models with dicom images
# * Normalize, transform

# Preprocessing to-do:
# - ~~Train/Validation/Test split~~
#     - ~~Make sure to keep class distribution~~
#     - ~~Split by patient, not image~~
#     
# * ~~Create random smaller subsets to get the model working~~
#     - ~~Class distribution, split by patient~~
#     - ~~Maybe try first with ~200 training? ~~
#     - ~~Once debugged, scale up to 1000, 10000, whole dataset~~   
#    
# * ~~3 channel windowing~~
#     - using brain, subdural, soft tissue right now
#     - consider using bone window instead of soft tissue
# 
# * ~~Transformations/data augmentation~~
#     - ~~random horizontal flip~~
#     - ~~random vertical flip~~
#     - ~~random rotate (should be able to fill empty space with black)~~
#     - ~~tensor~~
#     - ~~normalize~~
#     - ~~resize? Depends on model I think~~
#     - If needed, could also try:
#         - random noise
#     - If there is class imbalance problem, also consider over-/undersampling
#         - this might not be necessary if use weighted loss
#         
# * ~~Create DataLoader~~

# In[2]:


# Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import PIL
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
from sklearn.model_selection import train_test_split, GroupShuffleSplit


# In[3]:


# set global path and device
dpath = '/rsna-intracranial-hemorrhage-detection/'
trpath = '/rsna-intracranial-hemorrhage-detection/stage_2_train/'

if torch.cuda.is_available:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Label Dataframe(s)

# In[4]:


# Reshape and add patient IDs
def reshape_df(df_path): 
    # Reshape
    df = pd.read_csv(df_path)
    df['Image ID'] = df['ID'].str[:12]
    df['subtype'] = df['ID'].str[13:]
    df = df.drop(['ID'], axis=1)
    df = df.pivot_table(index=['Image ID'], columns='subtype', values='Label').reset_index().rename_axis(None, axis=1)
    # Get patient IDs from dicom files
    id_dict = {}
    since = time.time()
    for i in list(df['Image ID']):
        img_dcm = pydicom.dcmread(trpath + i + '.dcm')
        pat_id = str(img_dcm.PatientID)
        id_dict[i] = pat_id
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    df['Patient ID'] = df['Image ID'].map(id_dict)
    
    df = df[['Image ID', 'Patient ID', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']]
    
    return df

# 'reshape_df' function takes FOREVER. Only run the below once and then save csv until able to optimize
# labs_df_path = dpath + 'stage_2_train.csv'
# df_labs = reshape_df(labs_df_path) 
# df_labs.to_csv('./data/stage_2_train_reshaped.csv'We , index=False)


# In[5]:


df_labs = pd.read_csv('./data/stage_2_train_reshaped.csv')
df_labs


# Split data into train, validate, and test by patient ID 

# In[6]:


# Create group splitters
gss_val = GroupShuffleSplit(n_splits=1, train_size=0.65, random_state=8)
gss_test = GroupShuffleSplit(n_splits=1, train_size=0.45/0.85, random_state=8)
    # Validation set
for train_idx, val_idx in gss_val.split(X=df_labs, y=None, groups=df_labs['Patient ID']):
    df_temp = df_labs.iloc[train_idx].reset_index().drop('index', axis=1)
    df_val = df_labs.iloc[val_idx].reset_index().drop('index', axis=1)

    # Train and test set
for train_idx, test_idx in gss_test.split(X=df_temp, y=None, groups=df_temp['Patient ID']):
    df_train = df_temp.iloc[train_idx].reset_index().drop('index', axis=1)
    df_test = df_temp.iloc[test_idx].reset_index().drop('index', axis=1)

# Remove corrupted file from test set
df_test = df_test[df_test['Image ID'] != 'ID_6431af929']
    
print('train obs: ', len(df_train), 'val obs: ', len(df_val), 'test obs: ', len(df_test), '\n')

# Check class distribution
for i in ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']:
    print(' Train % {}: '.format(i), '{:.2%}'.format(df_train[i].sum() / len(df_train)), '\n',
          'Validation % {}: '.format(i), '{:.2%}'.format(df_val[i].sum() / len(df_val)), '\n',
          'Test % {}: '.format(i), '{:.2%}'.format(df_test[i].sum() / len(df_test)), '\n')
    # Dataset seems to be large enough that class distribution is reasonably well balanced. 
    # Potentially stratify if results indicate that it's necessary.

# Export train and test data to csv
df_train.to_csv('./data/df_train.csv', index=False)
df_val.to_csv('./data/df_val.csv', index=False)
df_test.to_csv('./data/df_test.csv', index=False)


# Create data subsets for model tuning/debugging

# In[7]:


df = pd.read_csv('./data/df_train.csv')
_ , tr_random = train_test_split(df, test_size=0.15, random_state=33)
tr_random.to_csv('./data/train_sub.csv',index=False)
print('train obs: ', len(tr_random))

df = pd.read_csv('./data/df_val.csv')
_ , val_random = train_test_split(df, test_size=0.05, random_state=33)
val_random.to_csv('./data/val_sub.csv',index=False)
print('val obs: ', len(val_random))

# Check class distribution
for i in ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']:
    print(' Train % {}: '.format(i), '{:.2%}'.format(tr_random[i].sum() / len(tr_random)), '\n',
          'Validation % {}: '.format(i), '{:.2%}'.format(val_random[i].sum() / len(val_random)), '\n')
    # Seems good enough. Just for debugging


# Image Dataset Class

# In[8]:


# Define windowing functions
def window(img, center, width):
    img = img.copy()
    img_min = center - width // 2
    img_max = center + width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img

def window_levels(img):
    image1 = window(img, 40, 80) # brain
    image2 = window(img, 80, 200) # subdural
    image3 = window(img, 40, 380) # source says bone, but according to sources, this is soft tissue windowing
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    img = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)

    return img

# CT windowing sources
    # https://www.stepwards.com/?page_id=21646
    # https://radiopaedia.org/articles/windowing-ct?lang=us
    # https://www.kaggle.com/dcstang/see-like-a-radiologist-with-systematic-windowing
    # https://www.kaggle.com/allunia/rsna-ih-detection-eda


# In[9]:


# Transforms, may need to copy into train script
train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)), # not sure if need resize
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize([0.456, 0.456, 0.456], [0.224, 0.224, 0.224])
        ])

val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)), # not sure if need resize
        transforms.Normalize([0.456, 0.456, 0.456], [0.224, 0.224, 0.224])
        ])

# Dataset class, 3 channel
class RSNADataset_3chan(Dataset):
    """
    RSNA Intracranial Hemorrhage Dataset from: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data
    Using 3 channel windowing
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """Args:
            csv_file (string): Path to csv with image IDs and labels
            root_dir (string): Directory with images
            transform (callable, optional): Optional transform to be applied to sample
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_dcm = pydicom.dcmread(self.root_dir + self.data_frame.iloc[idx, 0] + '.dcm')
        img_id = str(img_dcm.SOPInstanceUID)
        img_raw = img_dcm.pixel_array

        # Image window data
        ##### slope
        if type(img_dcm.RescaleSlope) == pydicom.multival.MultiValue:
            slope = int(img_dcm.RescaleSlope[0])
        else:
            slope = int(img_dcm.RescaleSlope)
        ##### intercept
        if type(img_dcm.RescaleIntercept) == pydicom.multival.MultiValue:
            intercept = int(img_dcm.RescaleIntercept[0])
        else:
            intercept = int(img_dcm.RescaleIntercept)
        ##### center
        if type(img_dcm.WindowCenter) == pydicom.multival.MultiValue:
            center = int(img_dcm.WindowCenter[0])
        else:
            center = int(img_dcm.WindowCenter)  
        ##### width
        if type(img_dcm.WindowWidth) == pydicom.multival.MultiValue:
            width = int(img_dcm.WindowWidth[0])
        else:
            width = int(img_dcm.WindowWidth) 

        # Window image
        img = img_raw * slope + intercept
        img = window_levels(img)

        # Normalize
        # min_im = np.min(img)
        # max_im = np.max(img)
        # img = ((img - min_im) / (max_im - min_im + 1e-4))
        
        # Transforms
        if self.transform:
            img = self.transform(img)
        
        # Get label (one-hot, but really two-hot)
        label = self.data_frame.loc[idx,'epidural':'any'].to_numpy().astype(float)
        
        # To tensors
        label = torch.tensor(label).long()
        
        sample = {'image':img, 'class':label, 'img_id':img_id}
        
        return sample

# Make sure it works
labs_df_path = './data/stage_2_train_reshaped.csv'

sample = RSNADataset_3chan(labs_df_path, trpath, transform=train_transform)

im_ind = 1
print(sample[im_ind]['image'].shape)
print(sample[im_ind]['class'])
print(sample[im_ind]['img_id'])

plt.figure(figsize=(8,8))
plt.imshow(np.moveaxis(sample[im_ind]['image'].numpy(), 0, 2))


# In[10]:


# Dataset class, 1 channel
class RSNADataset_1chan(Dataset):
    """
    RSNA Intracranial Hemorrhage Dataset from: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data
    Using 3 channel windowing
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """Args:
            csv_file (string): Path to csv with image IDs and labels
            root_dir (string): Directory with images
            transform (callable, optional): Optional transform to be applied to sample
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        img_dcm = pydicom.dcmread(self.root_dir + self.data_frame.iloc[idx, 0] + '.dcm')
        img_id = str(img_dcm.SOPInstanceUID)
        img_raw = img_dcm.pixel_array

        # Image window data
        ##### slope
        if type(img_dcm.RescaleSlope) == pydicom.multival.MultiValue:
            slope = int(img_dcm.RescaleSlope[0])
        else:
            slope = int(img_dcm.RescaleSlope)
        ##### intercept
        if type(img_dcm.RescaleIntercept) == pydicom.multival.MultiValue:
            intercept = int(img_dcm.RescaleIntercept[0])
        else:
            intercept = int(img_dcm.RescaleIntercept)
        ##### center
        if type(img_dcm.WindowCenter) == pydicom.multival.MultiValue:
            center = int(img_dcm.WindowCenter[0])
        else:
            center = int(img_dcm.WindowCenter)  
        ##### width
        if type(img_dcm.WindowWidth) == pydicom.multival.MultiValue:
            width = int(img_dcm.WindowWidth[0])
        else:
            width = int(img_dcm.WindowWidth) 

        # Window image
        img = img_raw * slope + intercept
        img = window(img, center, width)

        # Normalize
        min_im = np.min(img)
        max_im = np.max(img)
        img = ((img - min_im) / (max_im - min_im + 1e-4))
        
        # Transforms
        img = np.expand_dims(img, axis=2)
        img = np.tile(img, (1, 1, 3)) # make 3 chan
        if self.transform:
            img = self.transform(img)
        
        # Get label (one-hot, but really two-hot)
        label = self.data_frame.loc[idx,'epidural':'any'].to_numpy().astype(float)
        
        # To tensors
        label = torch.tensor(label).long()
        
        sample = {'image':img, 'class':label, 'img_id':img_id}
        
        return sample

# Make sure it works
labs_df_path = './data/stage_2_train_reshaped.csv'

sample = RSNADataset_1chan(labs_df_path, trpath, transform=train_transform)

im_ind = 1
print(sample[im_ind]['image'].shape)
print(sample[im_ind]['class'])
print(sample[im_ind]['img_id'])

plt.figure(figsize=(8,8))
plt.imshow(np.moveaxis(sample[im_ind]['image'].numpy(), 0, 2))


# Create Dataloader

# In[11]:


# Dataloader, copy into train script
