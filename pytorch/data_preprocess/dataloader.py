import collections
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import VeRiDataset, VeRiTestDataset

def create_dataloader(opt, is_train, drop_last=True):
    """
    Return the dataloader according to the opt.
    """
    if opt.data_augment:
        transform = transforms.Compose([
            transforms.Resize((opt.resize_height, opt.resize_width)),     
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((opt.resize_height, opt.resize_width)),     
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    if is_train: 
        dataset = VeRiDataset(opt, transform, is_train)
        batch_size = opt.batch_p
    else:
        dataset = VeRiTestDataset(opt, transform)
        batch_size = opt.test_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=opt.num_workers, drop_last=drop_last)
    return dataloader, len(dataset)

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
