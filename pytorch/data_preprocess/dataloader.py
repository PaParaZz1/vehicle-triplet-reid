import collections
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import VeRiDataset

def create_dataloader(opt, is_train):
    """
    Return the dataloader according to the opt.
    """
    if opt.data_augment:
        transform = transforms.Compose([
            transforms.Resize((100, 100)),     
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((100, 100)),     
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    dataset = VeRiDataset(opt, transform, is_train)
    dataloader = DataLoader(dataset, batch_size=opt.batch_p, shuffle=is_train, num_workers=opt.num_workers, drop_last=True)
    return dataloader

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='test dataset.')
    parser.add_argument('--train_set',default='/mnt/lustre/niuyazhe/data/VeRi/VeRi_train.csv')
    parser.add_argument('--image_root',default='/mnt/lustre/niuyazhe/data/VeRi/')
    parser.add_argument('--batch_k',default=4)
    parser.add_argument('--batch_p',default=8)
    parser.add_argument('--num_workers',default=8)
    parser.add_argument('--data_augment',default=True)
    opt = parser.parse_args()
    
    dataloader = create_dataloader(opt, True)
    idx, (imgs, labels) = next(enumerate(dataloader))
    print(imgs.shape)
    print(labels.shape)
    print("test dataloader pass")
