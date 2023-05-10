import torch
import torchvision.transforms as transforms
from PIL import Image
from model import AlexNet
device = "cuda" if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

net = AlexNet().to(device)
net.load_state_dict(torch.load('./best_model.pth.pth'))
net.eval()

image = Image.open('.jpg')
image_tensor = transform(image).unsqueeze(0).to(device)
output = net(image_tensor)
_, predicted = torch.max(output.data, 1)
classes = ('卡车','汽车','狗','猫','船','蛙','飞机','马','鸟','鹿')
print('Predicted:', classes[predicted.item()])