
# Imports here
import json
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
    
# Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method

### /path/to/image
parser.add_argument('path_to_image', type = str, default = 'flowers/test/102/image_08004.jpg', help = '/path/to/image')
### indicate directory to get checkpoint
parser.add_argument('checkpoint_path', type = str, default = 'trained_model_checkpoint.pth', help = 'directory in which checkpoint is stored')
### setting topk
parser.add_argument('--topk', type = int, default = 5 , help = '5 best guest of flower name', dest='topk')   
### setting category names
parser.add_argument('--category_names', type=str, default ='cat_to_name.json' , help = 'names and category of flowers', dest='category_names')   
### setting gpu
parser.add_argument('--gpu', type=bool, default =False  , help = 'choose to work with GPU or not', dest='gpu') 

# retrieve arguments
args = parser.parse_args()

arg_path_to_image =  args.path_to_image
arg_checkpoint_path =  args.checkpoint_path
arg_topk =  args.topk
arg_category_names = args.category_names
arg_gpu = args.gpu

###setting or device to cuda or cpu
device = torch.device("cuda" if torch.cuda.is_available() and arg_gpu else "cpu")


# Label mapping
with open(arg_category_names, 'r') as f:
    cat_to_name = json.load(f)
    
#print(cat_to_name)  


# Loading the checkpoint
# Inference for classification
# a function to load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    #model = getattr(torchvision.models, checkpoint['arch'])(pretrained = True)
    model = models.vgg16(pretrained = True)
    # freeze our features parameters
    for param in model.parameters():
        param.requires_grades = False
        model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
   
    return model


# Input image processing
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
       
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    # Provide the target width and height of the image
    (width, height) = (256, 256)
    im_resized = im.resize((width, height))
    
    #2.crop out the center 224x224 portion of the image
    half_crop = (256-224)*0.5
    right, lower = im_resized.size
    im_crop = im_resized.crop((half_crop, half_crop, right - half_crop, lower - half_crop))
    
    #3. Convert Color channels of images from 0-255 to 0-1
    np_image = np.array(im_crop)/255
    
    #4.Normalize the image 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (np_image-mean)/std
    
    #5.Reorder dimensions for the color channel to be the first dimension 
    image = image.transpose(2,0,1)
    
    return image

    
#show image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    plt.title(title)
    ax.imshow(image)
    
    return ax

#test
#imshow(torch.from_numpy(process_image(arg_path_to_image)),title="flower")

#Predictive function
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    #process numpy image
    image = torch.from_numpy(process_image(image_path))
    model.to(device)
     
    model.eval()
    image.type(torch.FloatTensor)
    image = image.unsqueeze(0)
    image = image.float().to(device)
    log_ps = model(image)
    lin_ps = torch.exp(log_ps)
    top_p, top_classes = lin_ps.topk(topk)
    top_p = top_p.cpu().detach().numpy()[0]
    top_classes = top_classes.cpu().detach().numpy()[0]
    inv_class_to_idx = {value : key for key, value in model.class_to_idx.items() }
    top_classes = [inv_class_to_idx[i] for i in top_classes]
    classes_names = [cat_to_name[class_] for class_ in top_classes]
    
    return list(top_p), top_classes, classes_names

# Prediction
model=load_checkpoint(arg_checkpoint_path)
image_path = arg_path_to_image                              
predict(image_path, model, topk=arg_topk)
