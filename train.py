# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import argparse
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
#import helper
import matplotlib.pyplot as plt

# torchvision package consists of popular datasets,
# model architectures, and common image transformations for computer vision.
from torchvision import datasets, transforms, models
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
from PIL import Image

# Create Parse using ArgumentParser
parser = argparse.ArgumentParser()
     
# Create 7 command line arguments as mentioned above using add_argument() from ArguementParser method

### data directory
parser.add_argument('data_directory', type = str, default = 'flowers', help = 'path to the folder of flowers images')
### indicate directory to save the checkpoint
parser.add_argument('--save_dir', type = str, default = '', help = 'directory to save the checkpoint', dest='save_directory')
### setting architecture
parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Choosing architecture', dest='architecture')   
### setting learning rate
parser.add_argument('--learning_rate', type=float, default = 0.001 , help = 'learning rate', dest='learning_rate')   
### hidden units
parser.add_argument('--hidden_units', type=int, default = 512 , help = 'hidden units', dest='hidden_units')   
### epochs
parser.add_argument('--epochs', type=int, default =3  , help = 'number of epochs', dest='epochs') 
### setting gpu
parser.add_argument('--gpu', type=bool, default =False  , help = 'choose to work with GPU or not', dest='gpu') 

# retrieve arguments
args = parser.parse_args()

arg_data_dir =  args.data_directory
arg_save_dir =  args.save_directory
arg_architecture =  args.architecture
arg_lr = args.learning_rate
arg_hidden_units = args.hidden_units
arg_epochs = args.epochs
arg_gpu = args.gpu

#Loading the data
### data directories
data_dir = arg_data_dir 
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

### Definition of our transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406] , [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406] , [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406] , [0.229, 0.224, 0.225])])


####Load the datasets with ImageFolder
train_data = torchvision.datasets.ImageFolder(train_dir,transform=train_transforms)
valid_data = torchvision.datasets.ImageFolder(valid_dir,transform=valid_transforms)
test_data = torchvision.datasets.ImageFolder(test_dir,transform=test_transforms)

### Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle =True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32, shuffle =False)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle =False)

###setting or device to cuda or cpu
device = torch.device("cuda" if torch.cuda.is_available() and arg_gpu else "cpu")

#Building and training the classifier

### apply model to input
if arg_architecture == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg16(pretrained=True)

               
### freeze our features parameters
for param in model.parameters():
    param.requires_grades = False
    
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
    ('fcl', nn.Linear(25088, arg_hidden_units)), 
    ('relu', nn.ReLU()),
    ('do' , nn.Dropout(p=0.2)),
    ('fc2', nn.Linear(arg_hidden_units,102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
print(model)
               


###criterion
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=arg_lr)
model.to(device)

# Definition of a training function
def trainer_(epochs, steps, running_loss, print_every, training_dataset_loader, dataset_loader, _model):
    
    for epoch in range(epochs):
        
        for images, labels in training_dataset_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = _model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                _model.eval()
                dataset_loss = 0
                accuracy = 0
                
                for images, labels in dataset_loader:
                    images, labels = images.to(device), labels.to(device)
                    logps = _model(images)
                    loss = criterion(logps, labels)
                    dataset_loss += loss.item()
                    
                    #calculate our accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every: .3f}).."
                          f"Valid loss: {dataset_loss/len(dataset_loader): .3f}.."
                          f"Valid accuracy: {accuracy/len(dataset_loader):.3f}")
                    
                    running_loss = 0
                    _model.train()           
    
### model trainning
trainer_(epochs = arg_epochs, steps = 0, running_loss = 0, print_every = 60, training_dataset_loader = trainloader, dataset_loader = validloader, _model = model)

# Save the checkpoint   
###
model.class_to_idx = train_data.class_to_idx
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())
 
checkpoint = { 'epochs' : arg_epochs,
                'arch': arg_architecture,
               'classifier': model.classifier,
               'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
               'class_to_idx' : model.class_to_idx}

torch.save(checkpoint, 'trained_model_checkpoint.pth')               
               

  





