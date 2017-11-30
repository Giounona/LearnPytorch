from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import torch.optim as optim


num_epochs = 10
batch_size = 100
learning_rate = 0.001



class MNISTDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations and absolute image paths.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #create a list with image names
        tmp_df = pd.read_csv(csv_file)
        self.tmp_df=tmp_df
        self.img_names = tmp_df['im_path']
        self.labels=tmp_df['label']


       # assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(),
        self.transform = transform

    def __len__(self):
        return len(self.img_names.index)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        image = np.float32(np.asarray(Image.open(img_name)))
        #image = image.convert('RGB')
        #label = np.float32(np.asarray([self.labels[idx]]))
        label = self.labels[idx]
        #label = torch.from_numpy(np.asarray(label).reshape([1, 1]))
        #label = torch.from_numpy(np.asarray(label))

        #label = torch.from_numpy(label)
        
        image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)

        return image, label

transformations = transforms.Compose([transforms.ToTensor()])

dset_train = MNISTDataset('train_mnist.csv')#,transformations)
dset_test  = MNISTDataset('test_mnist.csv')#,transformations)


train_loader = DataLoader(dset_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=10 # 1 for CUDA
                         # pin_memory=True # CUDA only
                         )


test_loader = DataLoader(dset_test,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=10 # 1 for CUDA
                         # pin_memory=True # CUDA only
                         )

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



cnn = CNN()
#cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
num_epochs=100
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)#.cuda()
        labels = Variable(labels)#.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                   % (epoch + 1, num_epochs, i + 1, dset_train.__len__() // batch_size, loss.data[0]))



# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)#.cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')