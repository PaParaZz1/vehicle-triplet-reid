import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class VeRiTestset(data.Dataset):
    def __init__(self, image_root, test_set_file, transform=None):
        self.transform = transform
        self.image_root = image_root

        self.pids, self.fids = self.load_testset(test_set_file)        

    def load_testset(self, csv_file, fail_on_missing=True):
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
        pids, fids = dataset.T

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

    def fid2tensor(self, fid_sample):
        path = self.image_root + fid_sample
        image = self.loader(path)
        if not self.transform is None:
            image_tensor = self.transform(image)
        return image_tensor

    def __getitem__(self, idx):
        fid_sample = self.fids[idx]
        image = self.fid2tensor(fid_sample)
        label = self.pids[idx]
        return image, label

    def __len__(self):
        return len(self.fids)


if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader
    image_root = '/home/cgy/server_223/Dataset/VeRi-776/'
    test_set_file = '/home/cgy/server_223/Dataset/VeRi-776/VeRi_test.csv'
    transform = transforms.Compose([
        transforms.Resize((288, 144)),     
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    test_set = VeRiTestset(image_root, test_set_file, transform=transform)
    dataloader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=8, drop_last=False)
    for img, label in dataloader:
        print(img.shape)
        print(label)
    print("test dataset pass")
