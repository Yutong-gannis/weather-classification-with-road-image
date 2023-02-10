import torch
import cv2
from albumentations import *
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
import torchvision


transforms = {
        x: Compose([
            Resize(224, 224),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_REPLICATE),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            OneOf([
                GaussianBlur(),
                GaussNoise(),
            ], p=0.2),

            Normalize(),
            ToTensorV2()
        ]) if x == 'train' else Compose([
            Resize(224, 224),

            Normalize(),
            ToTensorV2()
        ]) for x in ['train', 'test']
    }


def load_weather(device):
    path = r"D:\autodrive\Perception\SceneClassifier\weights\mobilenet_v3.pt"
    #model = EfficientNet.from_pretrained('efficientnet-b0')
    model = torchvision.models.mobilenet_v3_small(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def weather_infer(img, model):
    weather = ['cloudy', 'rainy', 'sunny', 'snowy', 'foggy']
    img = transforms['test'](image=img)['image'].unsqueeze(0).to('cuda')
    output = model(img)
    output = torch.argmax(output, dim=1).cpu().detach().numpy()
    if output[0] > 5:
        output[0] = 5
    output = weather[output[0]]
    return output



torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
weather = ['cloudy', 'rainy', 'sunny', 'snowy', 'foggy']
device = 'cuda:0'
img = cv2.imread(r"D:\autodrive\assets\test3_Moment_small.jpg")
model = load_weather(device)
output = weather_infer(img, model)
print(output)
