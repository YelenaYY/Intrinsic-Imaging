import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision import transforms
import numpy as np
import os

class IntrinsicDataset(Dataset):
    def __init__(self, dataset_paths, light_path):
        self.data = []
        self.light = np.load(light_path)
        for dataset_path in dataset_paths:
            self.data.extend(self._load_one_dataset(dataset_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_set = self.data[idx]

        try:
            mask = decode_image(img_set['mask'], mode='RGB') / 255.0
            reflectance = decode_image(img_set['reflectance'], mode='RGB') / 255.0
            shading = decode_image(img_set['shading'], mode='RGB') / 255.0
            # element-wise multiply
            reconstructed = reflectance * shading

            shading = shading[0, :, :] # take only the first channel

            normals = decode_image(img_set['normals'], mode='RGB') / 255.0
            
            # apply normalization to the normals
            normal_transform = transforms.Lambda(lambda img: (
                mask := img.pow(2).sum(0, keepdim=True) > 0.01,
                img.where(~mask.expand_as(img), (img - 0.5) * 2.0)
            )[1])
            normals = normal_transform(normals)

            depth = decode_image(img_set['depth'], mode='RGB') / 255.0
            depth = 1 - depth[0, :, :] # invert the depth
            depth = depth.unsqueeze(0)


            specular = decode_image(img_set['specular'], mode='RGB') / 255.0

            lights = torch.from_numpy(self.light[idx, :]).to(torch.float32)

            composite = decode_image(img_set['composite'], mode='RGB') / 255.0
        except Exception as e:
            print(e)
            print(f"Invalid image from the set, consider removing the whole set: {img_set}")
            return None


        return mask, reconstructed, reflectance, shading, normals, depth,  specular, composite, lights


    def _load_one_dataset(self, dataset_path):
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
        return data



if __name__ == "__main__":
    dataset = IntrinsicDataset(dataset_paths=["datasets/output/motorbike_train"], light_path="datasets/arrays/shader.npy")
    print("len(dataset)", len(dataset))
    print("Showing images from first index")
    mask, reconstructed, reflectance, shading, normals, depth, specular, composite, lights = dataset[0]
    print("mask.shape", mask.shape, "\t"*3, "type", mask.dtype, "min", mask.min(), "max", mask.max())
    print("reconstructed.shape", reconstructed.shape, "\t"*3, "type", reconstructed.dtype, "min", reconstructed.min(), "max", reconstructed.max())
    print("reflectance.shape", reflectance.shape, "\t"*3, "type", reflectance.dtype, "min", reflectance.min(), "max", reflectance.max())
    print("shading.shape", shading.shape, "\t"*3, "type", shading.dtype, "min", shading.min(), "max", shading.max())
    print("normals.shape", normals.shape, "\t"*3, "type", normals.dtype, "min", normals.min(), "max", normals.max())
    print("depth.shape", depth.shape, "\t"*3, "type", depth.dtype, "min", depth.min(), "max", depth.max())
    print("specular.shape", specular.shape, "\t"*3, "type", specular.dtype, "min", specular.min(), "max", specular.max())
    print("lights.shape", lights.shape, "\t"*3, "type", lights.dtype, lights)

    assert torch.any(reconstructed != reflectance )

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2)
    axs[0,0].imshow(reconstructed.permute(1, 2, 0).numpy())
    axs[0,0].set_title("Reconstructed")
    axs[0,1].imshow(composite.permute(1, 2, 0).numpy())
    axs[0,1].set_title("Composite")
    axs[1,0].imshow(reflectance.permute(1, 2, 0).numpy())
    axs[1,0].set_title("Reflectance")
    axs[1,1].imshow(shading.repeat(3, 1, 1).permute(1, 2, 0).numpy())
    axs[1,1].set_title("Shading")
    plt.show()


