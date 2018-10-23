import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import common


class VeRiDataset(data.Dataset):
    def __init__(self, opt, transform=None, train=True):

        self.transform = transform
        self.train = train

        if self.train:
            self.pids, self.fids = common.load_dataset(opt.train_set, opt.image_root)
        else:
            self.pids, self.fids = common.load_dataset(opt.test_set, opt.image_root)
        self.max_fid_len = max(map(len, self.fids))
        
        self.image_root = opt.image_root
        self.unique_pids = np.unique(self.pids)
        self.batch_k = opt.batch_k

    def sample_k_fids_for_pid(self, pid_item):
        mask = self.pids == pid_item
        possible_fids = self.fids[mask]
        
        count = possible_fids.shape[0]
        padding_count = (int)(np.ceil(self.batch_k/count)) * count
        full_range = np.array([x%count for x in range(padding_count)])
        np.random.shuffle(full_range)
        flag = full_range[:self.batch_k]
        selected_fids = possible_fids[flag]

        return selected_fids

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def fid2tensor(self, fid_sample):
        images = []
        for item in fid_sample:
            path = self.image_root + item
            image = self.loader(path)
            if not self.transform is None:
                image_tensor = self.transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            images.append(image_tensor)
        return torch.cat(images, dim=0)

    def __getitem__(self, idx):
        item = self.unique_pids[idx]
        fid_sample = self.sample_k_fids_for_pid(item)
        images = self.fid2tensor(fid_sample)
        item_np = np.asarray(int(item)).repeat(self.batch_k)
        item = torch.from_numpy(item_np)

        return images, item

    def __len__(self):
        return len(self.unique_pids)


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from argparse import ArgumentParser
    parser = ArgumentParser(description='test dataset.')
    parser.add_argument('--train_set',default='/mnt/lustre/niuyazhe/data/VeRi/VeRi_train.csv')
    parser.add_argument('--image_root',default='/mnt/lustre/niuyazhe/data/VeRi/')
    parser.add_argument('--batch_k',default=4)
    opt = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.Resize((100, 100)),       
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_set = VeRiDataset(opt, train=True, transform=transform)
    img, label = train_set[1]
    print(img.shape)
    print(label)
    print(img[0])
    print("test dataset pass")
