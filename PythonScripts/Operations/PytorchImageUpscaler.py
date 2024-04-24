import gc
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


class pytorchUpscaler:
    Labels = ["Pytorch-модель"]

    def UpcsalerInference(self, imgpaths, size):
        model = Upscaler(input_size=size)
        model.load_state_dict(torch.load(f=os.path.join(os.getcwd(), 'Models', 'Pytorch', 'bestModel.pt')))
        model.eval()

        # Define the preprocessing transformations
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for i in imgpaths:
            # Open the input low-resolution image
            input_image = Image.open(i)

            # Preprocess the input image
            input_tensor = preprocess(input_image).unsqueeze(0)

            # Upscale the input image
            with torch.no_grad():
                output_tensor = model(input_tensor)

            # # Postprocess the output tensor
            # output_tensor = output_tensor.squeeze(0)
            # # output_tensor = output_tensor.permute(1, 2, 0)  # CHW to HWC
            # output_image = transforms.ToPILImage()(output_tensor)
            #
            # # output_image = Image.fromarray(np.flip(output_image, -1))
            # # Save the upscaled image
            # output_image.save(i)

            save_image(output_tensor, i)


class Upscaler(nn.Module):

    def __init__(self, input_size):
        super(Upscaler, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Progressive Decoder
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2.0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2.0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.upsample1(x)
        x = self.relu(self.conv3(x))
        x = self.upsample2(x)
        x = self.relu(self.conv4(x))
        x = self.conv6(x)
        return x

    # def __init__(self, input_size):
    #     super(Upscaler, self).__init__()
    #     self.input_size = input_size
    #
    #     # Encoder
    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    #     self.pool = nn.MaxPool2d(2, 2)
    #
    #     # Decoder
    #     self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)
    #     self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    #     self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    #     self.conv5 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    #
    #     self.relu = nn.ReLU()
    #
    # def forward(self, x):
    #     # Encoder
    #     x = self.relu(self.conv1(x))
    #     x = self.relu(self.conv2(x))
    #     x = self.pool(x)
    #
    #     # Decoder
    #     x = self.relu(self.upsample(x))
    #     x = self.relu(self.upsample(x))
    #     x = self.relu(self.upsample(x))
    #     # x = self.relu(self.conv3(x))
    #     # x = self.upsample(x)
    #     # x = self.relu(self.conv4(x))
    #     # x = self.upsample(x)
    #     # x = self.conv5(x)
    #
    #     return x

    def train_model(model, train_loader, val_loader, num_epochs, device):
        """
        Trains the Upscaler model using the given training and validation data.

        Args:
            model (torch.nn.Module): The Upscaler model to be trained.
            train_loader (torch.utils.data.DataLoader): The training data loader.
            val_loader (torch.utils.data.DataLoader): The validation data loader.
            num_epochs (int): The number of training epochs.
            device (torch.device): The device (CPU or GPU) to use for training.
        """
        # Move the model to the device
        model.load_state_dict(torch.load("/kaggle/working/modelDict.pt"))
        model.to(device)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for high_res, trash in train_loader:
                print(high_res.size())

                low_res = model.getLowRes(high_res)

                high_res = high_res.to(device)
                low_res = low_res.to(device)

                # Forward pass
                output = model(low_res)
                loss = criterion(output, high_res)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                gc.collect()
                torch.cuda.empty_cache()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for high_res, trash in val_loader:
                    low_res = model.getLowRes(high_res)

                    high_res = high_res.to(device)
                    low_res = low_res.to(device)

                    output = model(low_res)
                    val_loss += criterion(output, high_res).item()
                    gc.collect()
                    torch.cuda.empty_cache()

            val_loss /= len(val_loader)
            # Print the progress
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save a sample output image
            torch.save(model.state_dict(), "modelDict.pt")
            sample_output = model(low_res[:1])
            save_image(sample_output, f'sample_output_epoch_{epoch + 1}.png')

        return model

    def getLowRes(self, high_res):
        img_array = high_res.numpy()
        # Transpose to (32, 256, 256, 3) as OpenCV uses HWC format
        img_array = np.transpose(img_array, (0, 2, 3, 1))
        # Resize each image in the batch
        resized_images = []
        for img in img_array:
            resized_img = cv2.resize(img, (400, 400))  # Resize to (64, 64)
            resized_images.append(resized_img)
        # Convert resized images back to numpy array
        resized_images = np.array(resized_images)
        # Convert back to tensor
        resized_tensor = torch.from_numpy(resized_images)
        # Transpose back to tensor shape (32, 3, 64, 64)
        return resized_tensor.permute(0, 3, 1, 2)

    def UpcsalerInference(self):
        model = Upscaler()
        model.load_state_dict(torch.load('upscaler_model.pth'))
        model.eval()

        # Define the preprocessing transformations
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Open the input low-resolution image
        input_image = Image.open('input_image.jpg')

        # Preprocess the input image
        input_tensor = preprocess(input_image).unsqueeze(0)

        # Upscale the input image
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Postprocess the output tensor
        output_tensor = output_tensor.squeeze(0)
        output_tensor = output_tensor.permute(1, 2, 0)  # CHW to HWC
        output_image = transforms.ToPILImage()(output_tensor)

        # Save the upscaled image
        output_image.save('upscaled_image.jpg')
