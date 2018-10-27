import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class VeRiDataset(data.Dataset):
    def __init__(self, opt, transform=None, train=True):

        self.transform = transform
        self.train = train
        self.image_root = opt.image_root
        self.batch_k = opt.batch_k

        if self.train:
            self.pids, self.fids = self.load_dataset(opt.train_set, opt.image_root)
        else:
            self.pids, self.fids = self.load_dataset(opt.test_set, opt.image_root)
        self.max_fid_len = max(map(len, self.fids))
        
        self.unique_pids = np.unique(self.pids)

    def load_dataset(self, csv_file, fail_on_missing=True):
        """ Loads a dataset .csv file, returning PIDs and FIDs.

        PIDs are the "person IDs", i.e. class names/labels.
        FIDs are the "file IDs", which are individual relative filenames.

        Args:
            csv_file (string, file-like object): The csv data file to load.
            image_root (string): The path to which the image files as stored in the
                csv file are relative to. Used for verification purposes.
                If this is `None`, no verification at all is made.
            fail_on_missing (bool or None): If one or more files from the dataset
                are not present in the `image_root`, either raise an IOError (if
                True) or remove it from the returned dataset (if False).

        Returns:
            (pids, fids) a tuple of numpy string arrays corresponding to the PIDs,
            i.e. the identities/classes/labels and the FIDs, i.e. the filenames.

        Raises:
            IOError if any one file is missing and `fail_on_missing` is True.
        """
        dataset = np.genfromtxt(csv_file, delimiter=',', dtype='|U')
        pids, fids= dataset.T

        # Possibly check if all files exist
        if self.image_root is not None:
            missing = np.full(len(fids), False, dtype=bool)
            for i, fid in enumerate(fids):
                missing[i] = not os.path.isfile(os.path.join(self.image_root, fid))

            missing_count = np.sum(missing)
            if missing_count > 0:
                if fail_on_missing:
                    raise IOError('Using the `{}` file and `{}` as an image root {}/'
                                '{} images are missing'.format(
                                    csv_file, self.image_root, missing_count, len(fids)))
                else:
                    print('[Warning] removing {} missing file(s) from the'
                        ' dataset.'.format(missing_count))
                    # We simply remove the missing files.
                    fids = fids[np.logical_not(missing)]
                    pids = pids[np.logical_not(missing)]
        return pids, fids
                
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

class VeRiTestDataset(data.Dataset):
    def __init__(self, opt, transform=None, train=True):
        self.crop_margin = 8
        self.transform = transform
        self.image_root = opt.image_root

        self.pids, self.fids = self.load_dataset(opt.test_set, opt.image_root)
        self.max_fid_len = max(map(len, self.fids))
        

    def load_dataset(self, csv_file, fail_on_missing=True):
        """ Loads a dataset .csv file, returning PIDs and FIDs.

        PIDs are the "person IDs", i.e. class names/labels.
        FIDs are the "file IDs", which are individual relative filenames.

        Args:
            csv_file (string, file-like object): The csv data file to load.
            image_root (string): The path to which the image files as stored in the
                csv file are relative to. Used for verification purposes.
                If this is `None`, no verification at all is made.
            fail_on_missing (bool or None): If one or more files from the dataset
                are not present in the `image_root`, either raise an IOError (if
                True) or remove it from the returned dataset (if False).

        Returns:
            (pids, fids) a tuple of numpy string arrays corresponding to the PIDs,
            i.e. the identities/classes/labels and the FIDs, i.e. the filenames.

        Raises:
            IOError if any one file is missing and `fail_on_missing` is True.
        """
        dataset = np.genfromtxt(csv_file, delimiter=',', dtype='|U')
        pids, fids= dataset.T

        # Possibly check if all files exist
        if self.image_root is not None:
            missing = np.full(len(fids), False, dtype=bool)
            for i, fid in enumerate(fids):
                missing[i] = not os.path.isfile(os.path.join(self.image_root, fid))

            missing_count = np.sum(missing)
            if missing_count > 0:
                if fail_on_missing:
                    raise IOError('Using the `{}` file and `{}` as an image root {}/'
                                '{} images are missing'.format(
                                    csv_file, self.image_root, missing_count, len(fids)))
                else:
                    print('[Warning] removing {} missing file(s) from the'
                        ' dataset.'.format(missing_count))
                    # We simply remove the missing files.
                    fids = fids[np.logical_not(missing)]
                    pids = pids[np.logical_not(missing)]
        return pids, fids

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def fid2tensor(self, fid):
        path = self.image_root + fid
        image = self.loader(path)
        if not self.transform is None:
            image_tensor = self.transform(image)
        else:
            raise NotImplementedError("not implemented totensor op")
        return image_tensor

    def my_flip(self, image):
        image_flip = image.numpy()
        result = np.flip(image_flip, axis=2).copy()
        return torch.from_numpy(result)

    def my_crop(self, image_in):
        c, h, w = image_in.shape
        margin = self.crop_margin
        image = image_in.numpy()
        image = image.transpose(1,2,0)
        image = cv2.resize(image, (w+2*margin, h+2*margin))
        image = image.transpose(2,0,1)
        center = image[:, margin:-margin, margin:-margin]
        center = torch.from_numpy(center)
        
        top_left = image[:, 0:-2*margin, 0:-2*margin]
        top_left = torch.from_numpy(top_left)

        top_right = image[:, 0:-2*margin, 2*margin:]
        top_right = torch.from_numpy(top_right)

        bottom_left = image[:, 2*margin:, 0:-2*margin]
        bottom_left = torch.from_numpy(bottom_left)

        bottom_right = image[:, 2*margin:, 2*margin:]
        bottom_right = torch.from_numpy(bottom_right)

        return center, top_left, top_right, bottom_left, bottom_right
        
    def __getitem__(self, idx):
        item = self.fids[idx]
        image = self.fid2tensor(item)
        image_flip = self.my_flip(image)
        center, top_left, top_right, bottom_left, bottom_right = self.my_crop(image)
        return torch.stack([image, image_flip, center, top_left, top_right, bottom_left, bottom_right], dim=0)

    def __len__(self):
        return self.max_fid_len


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
