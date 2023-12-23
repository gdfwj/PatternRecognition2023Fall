import cv2
from PIL import Image
from torchvision import transforms

# 1、转灰度图
transform = transforms.Compose([
        transforms.Resize((128,128)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    )
image = "jer_2.1.jpg"
image = Image.open(image).convert('RGB')
image = transforms.ToTensor()(image).numpy()
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray", gray)