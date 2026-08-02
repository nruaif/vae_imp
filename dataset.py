import webdataset as wds
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import io
from PIL import Image

def warn_and_continue(exn):
    print(f"Warning: {exn}")
    return True

class WDSLoader:
    def __init__(self, url, csv_path=None, image_size=64, batch_size=16, num_workers=4):
        self.url = url
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ratio = (3. / 4., 4. / 3.)

    def preprocess(self, sample):
        image_key = None
        for key in ["image", "jpg", "jpeg", "png", "webp"]:
            if key in sample:
                image_key = key
                break

        if image_key is None:
            return None

        try:
            image_bytes = sample[image_key]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

        # Global Crop (Scale 0.5 to 1.0)
        i_g, j_g, h_g, w_g = transforms.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=self.ratio)
        image_global = F.resized_crop(image, i_g, j_g, h_g, w_g, size=(self.image_size, self.image_size))

        # Local Crop (Scale 0.05 to 0.5)
        i_l, j_l, h_l, w_l = transforms.RandomResizedCrop.get_params(image, scale=(0.05, 0.5), ratio=self.ratio)
        image_local = F.resized_crop(image, i_l, j_l, h_l, w_l, size=(self.image_size, self.image_size))

        # To Tensor and Normalize [-1, 1]
        image_global = (F.to_tensor(image_global) - 0.5) * 2.0
        image_local = (F.to_tensor(image_local) - 0.5) * 2.0

        return {
            "image_global": image_global,
            "image_local": image_local
        }

    def make_loader(self):
        dataset = (
            wds.WebDataset(self.url, nodesplitter=wds.split_by_node, handler=warn_and_continue)
            .shuffle(1000)
            .map(self.preprocess, handler=warn_and_continue)
            .select(lambda x: x is not None)
            .to_tuple("image_global", "image_local", handler=warn_and_continue)
            .batched(self.batch_size, partial=False)
        )

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader