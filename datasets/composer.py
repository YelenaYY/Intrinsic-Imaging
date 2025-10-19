from torch.utils.data import Dataset
from .base import IntrinsicDataset
import random

class ComposerDataset(Dataset):
    def __init__(self, labeled_dataset_paths, unlabeled_dataset_paths, light_path, max_num_images_per_dataset=None, random_seed=None, cache=False):
        self.labeled_dataset_paths = labeled_dataset_paths
        self.unlabeled_dataset_paths = unlabeled_dataset_paths
        self.light_path = light_path
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)

        self.labeled_dataset = IntrinsicDataset(labeled_dataset_paths, light_path, max_num_images_per_dataset, cache)
        self.unlabeled_dataset = IntrinsicDataset(unlabeled_dataset_paths, light_path, max_num_images_per_dataset, cache)
        print("len(labeled_dataset)", len(self.labeled_dataset))
        print("len(unlabeled_dataset)", len(self.unlabeled_dataset))

    def __len__(self):
        return len(self.unlabeled_dataset)

    def __getitem__(self, idx):
        unlabeled_img_set = self.unlabeled_dataset[idx]

        # randomly sample a labeled image
        labeled_idx = random.randint(0, len(self.labeled_dataset) - 1)
        labeled_img_set = self.labeled_dataset[labeled_idx]

        return (labeled_img_set, unlabeled_img_set)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import tqdm

    labeled_dataset_paths = ["datasets/output/cone", "datasets/output/cylinder", "datasets/output/sphere", "datasets/output/torus", "datasets/output/cube"]
    unlabeled_dataset_paths = ["datasets/output/suzanne_train", "datasets/output/teapot_train", "datasets/output/bunny_train"]

    dataset = ComposerDataset(labeled_dataset_paths, unlabeled_dataset_paths, "datasets/arrays/shader.npy", cache=True)

    loader = DataLoader(dataset, batch_size=4, num_workers=24, shuffle=True, pin_memory=True)
    for i in range(2):
        for batch in tqdm.tqdm(loader, total=len(loader)):
            labeled_img_set, unlabeled_img_set = batch
            mask, reconstructed, reflectance, shading, normals, depth,  specular, composite, lights = labeled_img_set

            mask, reconstructed, reflectance, shading, normals, depth,  specular, composite, lights = unlabeled_img_set


    # print(len(dataset))
    # labeled_img_set, unlabeled_img_set = dataset[0]
    # mask, reconstructed, reflectance, shading, normals, depth,  specular, composite, lights = labeled_img_set
    # print("mask.shape", mask.shape, "\t"*3, "type", mask.dtype, "min", mask.min(), "max", mask.max())


    # mask, reconstructed, reflectance, shading, normals, depth,  specular, composite, lights = unlabeled_img_set
    # print("mask.shape", mask.shape, "\t"*3, "type", mask.dtype, "min", mask.min(), "max", mask.max())