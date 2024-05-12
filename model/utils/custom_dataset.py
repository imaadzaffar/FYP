import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir, model_arch, mask_type="right", transform=False):
        self.data_dir = data_dir
        self.model_arch = model_arch
        self.transform = transform

        # image dir
        self.image_dir = os.path.join(data_dir, "images_best_slice")
        self.image_files = sorted(os.listdir(self.image_dir))

        # mask dir
        if mask_type == "both":
            self.mask_dir = os.path.join(data_dir, "masks_both")
        elif mask_type == "right":
            self.mask_dir = os.path.join(data_dir, "masks_right")
        self.mask_files = sorted(os.listdir(self.mask_dir))

        assert len(self.image_files) == len(self.mask_files)

    def slice_transform(self, slice_image):
        if self.model_arch == "brain_mri":
            m, s = np.mean(slice_image, axis=(0, 1)), np.std(slice_image, axis=(0, 1))
            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop((256, 256)),
                    transforms.Normalize(mean=m, std=s),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        input_tensor = preprocess(slice_image)
        return input_tensor

    def mask_transform(self, mask_image):
        if self.model_arch == "brain_mri":
            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.CenterCrop((256, 256)),
                ]
            )
        input_tensor = preprocess(mask_image)
        return input_tensor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # transform
        if self.transform:
            image = self.slice_transform(image)
            mask = self.mask_transform(mask)

        return image, mask

