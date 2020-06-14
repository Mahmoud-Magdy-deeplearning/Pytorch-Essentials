import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Create your own dataset object

class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        
        # Image directory
        self.data_dir=data_dir
        
        # The transform is goint to be used on image
        self.transform = transform
        
        # Load the CSV file contians image info
        self.data_name= pd.read_csv(csv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0] 
    
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        
        # Image file path
        img_name=self.data_dir + self.data_name.iloc[idx, 2]
        
        # Open image file
        image = Image.open(img_name)
        
        
        # The class label for the image
        y = self.data_name.iloc[idx, 3]
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
        
        
        
# Create the dataset objects
train_dataset = Dataset(csv_file=train_csv_file
                        , data_dir='/resources/data/training_data_pytorch/')

# Create the dataloader objects
dataloader = DataLoader(train_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

# looping through epochs then looping through batches then looping through dataset single batch then do whatever you want
epochs=input()
iterator = iter(dataloader)
for i in range(epochs):
    for j in range(4):
        images, labels = next(iterator)
