import numpy as np
import argparse
import os
import torch

from torchvision import transforms
from PIL import Image

from model import VGG as CNN
#from model import customCNN as CNN

parser = argparse.ArgumentParser(description='Test Example')
parser.add_argument('--imagePath',type=str,default='test/')
parser.add_argument('--rootDir', type=str, default='Data/', help='Directory path to root of the training data')
parser.add_argument('--checkpoint',type=str,default='checkpoint/best_vgg.pth')


opt = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

idx_to_classes = {index: cavity for index, cavity in enumerate(os.walk(opt.rootDir).__next__()[1])}
transformation = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])


for file in os.listdir('test'):
    image = transformation(Image.open(os.path.join(opt.imagePath,file)).convert("RGB")).to(device).unsqueeze(0)

    model = CNN(num_classes=len(idx_to_classes)).to(device)
    model.load_state_dict(torch.load(opt.checkpoint, map_location=lambda storage, loc: storage))

    _, predict = torch.max(model(image).data.cpu(), 1)
    print("It's a {}!".format(idx_to_classes[predict.item()]),file)


