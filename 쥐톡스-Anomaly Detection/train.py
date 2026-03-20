import pandas as pd
import torch.nn as nn
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from torch import optim
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (44, 33)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(16, 32, (40, 30)),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 64, (12, 9)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(64, 128, (4, 3)),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=3, mode="nearest"),

            nn.ConvTranspose2d(64, 32, (12, 9)),
            nn.ReLU(),
            nn.Upsample(scale_factor=3, mode="nearest"),

            nn.ConvTranspose2d(32, 16, (40, 30)),
            nn.ReLU(),
            nn.Upsample(scale_factor=3, mode="nearest"),
            nn.ConvTranspose2d(16, 3, (44, 33)),
            nn.Sigmoid()
        )


    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
            # ★ 어떤 입력 크기든 출력 크기를 입력과 동일하게 강제
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(out, size=x.shape[-2:], mode="nearest")

        return out
    

class Dataset_train(Dataset):
    def __init__(self, root_dir, transform=None): 
        self.image_paths = glob(os.path.join(root_dir, "*", "*", "*.png"))
        self.image_paths.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_path 
    
class Dataset_eval(Dataset):
    def __init__(self, root_dir, transform=None): 
        self.image_paths = glob(os.path.join(root_dir, "*", "*", "*.png"))
        self.image_paths.sort()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_path 

    
# tensor -> numpy -> HWC 변환 함수
def imshow_tensor(img_tensor):
    img = img_tensor.detach().cpu().numpy()  # [C, H, W]
    img = img.transpose(1, 2, 0)  # [H, W, C]
    plt.imshow(img, cmap='gray' if img.shape[2]==1 else None)
    plt.axis('off')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)


train = '/root/hdd/yeonseo/ess/8_drop_frequency(3.3)/train'
val_normal = '/root/hdd/yeonseo/ess/8_drop_frequency(3.3)/val/normal'
val_abnormal = '/root/hdd/yeonseo/ess/8_drop_frequency(3.3)/val/abnormal'
transform = ToTensor()

train_folder = Dataset_train(root_dir=train, transform=transform)
normal_folder = Dataset_eval(root_dir=val_normal, transform=transform)
abnormal_folder = Dataset_eval(root_dir=val_abnormal, transform=transform)

train_loader = DataLoader(train_folder, batch_size=50, shuffle=True)
normal_loader = DataLoader(normal_folder, batch_size=50, shuffle=False)
abnormal_loader = DataLoader(abnormal_folder, batch_size=50, shuffle=False)

# 모델, 옵티마이저, criterion 설정
model = ConvAutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

model_path = '/root/ssd/yeonseo/ess/OCC_Algorithm/model'
os.makedirs(model_path, exist_ok=True)

steps = 0
total_steps = len(train_loader)
num_epoch = 30

running_loss = 0.0
for epoch in range(num_epoch):
    model.train()
    for i, (inputs, paths) in enumerate(train_loader, 0):
        inputs = inputs.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if epoch % 3 == 1:
            plt.figure(figsize=(6, 3))

            # input
            plt.subplot(1, 2, 1)
            imshow_tensor(inputs[0]) # batch에서 첫 번째만 표시
            plt.title("Input")

            # output
            plt.subplot(1, 2, 2)
            imshow_tensor(outputs[0])
            plt.title("Output")

            plt.show()
            plt.close()
            print(f"{epoch} epoch training loss: {running_loss/30}")
            running_loss = 0.0
    lr_sche.step()

    save_path = os.path.join(model_path, f"epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    normal_loss = 0.0
    with torch.no_grad():
        for inputs, _ in normal_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            normal_loss += loss.item() * inputs.size(0)

    normal_loss /= len(normal_loader.dataset)
    print(f"{epoch+1} epoch validation normal loss: {normal_loss}")
    model.eval()
    abnormal_loss = 0.0
    with torch.no_grad():
        for inputs, _ in abnormal_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            abnormal_loss += loss.item() * inputs.size(0)
            
            plt.figure(figsize=(6, 3))
            
            # input
            plt.subplot(1, 2, 1)
            imshow_tensor(inputs[0])  # batch에서 첫 번째만 표시
            plt.title("Input")

            # output
            plt.subplot(1, 2, 2)
            imshow_tensor(outputs[0])
            plt.title("Output")
            plt_save = "/".join(_[0].split("/")[-3:])
            save_path = os.path.join("/root/hdd/yeonseo/ess/OCC_validation", f"epoch_{epoch+1}", os.path.dirname(plt_save))
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, plt_save.split("/")[-1]))
            plt.close()

    abnormal_loss /= len(abnormal_loader.dataset)
    print(f"{epoch+1} epoch validation abnormal loss: {abnormal_loss}")