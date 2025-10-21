import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
from torchvision import transforms
import numpy as np
import os
import traceback
class IntrinsicDataset(Dataset):
    def __init__(self, dataset_paths, light_path, max_num_images_per_dataset=None, cache=False):
        self.data = []
        self.light = np.load(light_path)
        self.dataset_sizes = []
        self.cache = cache
        self.cache_data = {}
        for dataset_path in dataset_paths:
            data = self._load_one_dataset(dataset_path, max_num_images_per_dataset)
            self.dataset_sizes.append(len(data))
            self.data.extend(data)

    def __len__(self):
        return len(self.data)
    
    def idx_to_light_idx(self, idx):
        current_position = 0
        
        for size in self.dataset_sizes:
            if idx < current_position + size:
                return idx - current_position
            current_position += size
        raise ValueError(f"Index {idx} is out of range")


    def __getitem__(self, idx):
        if self.cache and idx in self.cache_data:
            return self.cache_data[idx]

        img_set = self.data[idx]

        try:
            mask = decode_image(img_set['mask'], mode=ImageReadMode.RGB) / 255.0
            reflectance = decode_image(img_set['reflectance'], mode=ImageReadMode.RGB) / 255.0
            shading = decode_image(img_set['shading'], mode=ImageReadMode.RGB) / 255.0
            # element-wise multiply
            reconstructed = reflectance * shading

            shading = shading[0, :, :] # take only the first channel
            shading = shading.unsqueeze(0)

            normals = decode_image(img_set['normals'], mode=ImageReadMode.RGB) / 255.0
            
            # apply normalization to the normals
            normal_transform = transforms.Lambda(lambda img: (
                mask := img.pow(2).sum(0, keepdim=True) > 0.01,
                img.where(~mask.expand_as(img), (img - 0.5) * 2.0)
            )[1])
            normals= normal_transform(normals)

            depth = decode_image(img_set['depth'], mode=ImageReadMode.RGB) / 255.0
            depth = 1 - depth[0, :, :] # invert the depth
            depth = depth.unsqueeze(0)


            specular = decode_image(img_set['specular'], mode=ImageReadMode.RGB) / 255.0

            light_idx = self.idx_to_light_idx(idx)
            lights = torch.from_numpy(self.light[light_idx, :]).to(torch.float32)

            composite = decode_image(img_set['composite'], mode=ImageReadMode.RGB) / 255.0
        except Exception as e:
            print("Error: ", traceback.format_exc())
            raise e

        if self.cache:
            self.cache_data[idx] = (mask, reconstructed, reflectance, shading, normals, depth,  specular, composite, lights)
        return mask, reconstructed, reflectance, shading, normals, depth,  specular, composite, lights


    def _load_one_dataset(self, dataset_path, max_num_images_per_dataset):
        data = []

        paths = os.listdir(dataset_path)

        index_set = set()
        for path in paths:
            # find the index in the filename before first _
            index = path.find('_')
            if index == -1:
                continue

            
            index = path[:index]
            if index.isdigit():
                index = int(index)
            else:
                continue

            if index not in index_set:
                index_set.add(index)

                new_data = {
                    'mask': os.path.join(dataset_path, f"{index}_mask.png"),
                    'reflectance': os.path.join(dataset_path, f"{index}_albedo.png"),
                    'composite': os.path.join(dataset_path, f"{index}_composite.png"),
                    'normals': os.path.join(dataset_path, f"{index}_normals.png"),
                    'depth': os.path.join(dataset_path, f"{index}_depth.png"),
                    'lights': os.path.join(dataset_path, f"{index}_lights.png"),
                    'shading': os.path.join(dataset_path, f"{index}_shading.png"),
                    'specular': os.path.join(dataset_path, f"{index}_specular.png")
                }
                assert os.path.exists(new_data['mask']), f"Mask image {new_data['mask']} does not exist"
                assert os.path.exists(new_data['reflectance']), f"Reflectance image {new_data['reflectance']} does not exist"
                assert os.path.exists(new_data['composite']), f"Composite image {new_data['composite']} does not exist"
                assert os.path.exists(new_data['normals']), f"Normals image {new_data['normals']} does not exist"
                assert os.path.exists(new_data['depth']), f"Depth image {new_data['depth']} does not exist"
                assert os.path.exists(new_data['lights']), f"Lights image {new_data['lights']} does not exist"
                assert os.path.exists(new_data['shading']), f"Shading image {new_data['shading']} does not exist"
                assert os.path.exists(new_data['specular']), f"Specular image {new_data['specular']} does not exist"
                data.append(new_data)
                if max_num_images_per_dataset is not None and len(data) >= max_num_images_per_dataset:
                    break
        index_set = list(index_set)
        index_set.sort()
        print("Successfully loaded dataset from", dataset_path, "with", len(data), "images", " idx range", index_set[0], "-", index_set[-1])
        return data



if __name__ == "__main__":
    dataset = IntrinsicDataset(dataset_paths=["datasets/output/motorbike_train"], light_path="datasets/arrays/shader.npy")
    print("len(dataset)", len(dataset))
    print("Showing images from first index")
    mask, reconstructed, reflectance, shading, normals, depth, specular, composite, lights = dataset[0] # type: ignore
    print("mask.shape", mask.shape, "\t"*3, "type", mask.dtype, "min", mask.min(), "max", mask.max())
    print("reconstructed.shape", reconstructed.shape, "\t"*3, "type", reconstructed.dtype, "min", reconstructed.min(), "max", reconstructed.max())
    print("reflectance.shape", reflectance.shape, "\t"*3, "type", reflectance.dtype, "min", reflectance.min(), "max", reflectance.max())
    print("shading.shape", shading.shape, "\t"*3, "type", shading.dtype, "min", shading.min(), "max", shading.max())
    print("normals.shape", normals.shape, "\t"*3, "type", normals.dtype, "min", normals.min(), "max", normals.max()) # type: ignore
    print("depth.shape", depth.shape, "\t"*3, "type", depth.dtype, "min", depth.min(), "max", depth.max())
    print("specular.shape", specular.shape, "\t"*3, "type", specular.dtype, "min", specular.min(), "max", specular.max())
    print("lights.shape", lights.shape, "\t"*3, "type", lights.dtype, lights)

    assert torch.any(reconstructed != reflectance )

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3)
    axs[0,0].imshow(reconstructed.permute(1, 2, 0).numpy())
    axs[0,0].set_title("Reconstructed")
    axs[0,1].imshow(composite.permute(1, 2, 0).numpy())
    axs[0,1].set_title("Composite")
    axs[1,0].imshow(reflectance.permute(1, 2, 0).numpy())
    axs[1,0].set_title("Reflectance")
    axs[1,1].imshow(shading.repeat(3, 1, 1).permute(1, 2, 0).numpy())
    axs[1,1].set_title("Shading")
    axs[2,0].imshow(normals.permute(1, 2, 0).numpy())
    axs[2,0].set_title("Normals")
    axs[2,1].imshow(depth.permute(1, 2, 0).numpy())
    axs[2,1].set_title("Depth")
    plt.show()

    dataset = IntrinsicDataset(dataset_paths=["datasets/output/cube", "datasets/output/cone", "datasets/output/cylinder", "datasets/output/sphere", "datasets/output/torus"], light_path="datasets/arrays/shader.npy")

    print("idx 10000 to light idx", dataset.idx_to_light_idx(10000))


