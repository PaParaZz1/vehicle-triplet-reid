#Readme

## requirements:
 - pytorch0.3.0

## code structure
 - data_preprocess
   - dataset
   - dataloader
 - backbone
   - resnet_interface(now support resnet50)
 - head
   - multi_branch_attention(both 'addition' and 'concat')
 - loss
   - batch_hard loss(now support 'Euclidean' and 'Square Euclidean' metric)
   - KLloss(kl divergence loss for attention mask)
 - modules.py (network sub-module)
 - common.py (command setting)
 - network.py (ReID network)
 - train.py (train code)

## tips
 - Due to modules.py is a old version, it uses tab as indent while others use 4 space
 - input normalization (3-channelwise, mean=0.5, std=0.5)
 - default use resize and RandomHorizontalFlip data augmentation
 - dataloader use 'drop_last' argument
 - almost network layers use 'xavier' initialization, only the last fc in head uses 'orthogonal' initialization
 - kl divergence calculates all the pairs of branch attention modules, default use size_average=False, not average 
   for element, but average for batch_size

