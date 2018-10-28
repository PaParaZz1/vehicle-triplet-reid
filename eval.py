import torch
import torchvision
from torchvision import datasets, transforms

import os
import numpy as np

from data_preprocess.dataset_test import VeRiTestset
from network import ReIDNetwork
from utils import extract_feature, evaluate, get_id


def main():

	"""
	Configs
	"""
	GPU_ID = 3
	BATCH_SIZE = 64
	IMG_SIZE = (288, 144)
	NET_TYPE = 'resnet50'
	WHICH = 'last'
	LOSS = 'proto'
	EXP_DIR = 'exp/{}/'.format(LOSS)
	normalize_feature = True
	single_gpu = False

	"""
	Dataset
	"""
	image_root = '/home/cgy/server_223/Dataset/VeRi-776/'
	dataset_file = {'gallery': '/home/cgy/server_223/Dataset/VeRi-776/VeRi_test.csv',
	                'query': '/home/cgy/server_223/Dataset/VeRi-776/VeRi_query.csv'}

	print('Generating dataset...')
	transform = transforms.Compose([
	    transforms.Resize((288, 144), interpolation=3),     
	    transforms.ToTensor(),
	    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	    ])
	datasets = {x: VeRiTestset(image_root, dataset_file[x], transform=transform) for x in ['gallery', 'query']}
	dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
		shuffle=False, num_workers=16, drop_last=False) for x in ['gallery', 'query']}
	print('Done.')

	"""
	Model
	"""
	print('Restoring ' + NET_TYPE + ' model...')
	if NET_TYPE == 'resnet50':
		model = ReIDNetwork(1000, NET_TYPE, 'MultiBranchAttention', 2048, 128,
	                        'concat', 5, is_backbone=True)
	else:
		raise NotImplementedError

	# single gpu
	if single_gpu:
		model.load_state_dict(torch.load('{}/checkpoint_{}.pth'.format(
			EXP_DIR, WHICH)))
	# multiple gpus
	else:
		state_dict = torch.load('{}/checkpoint_{}.pth'.format(
			EXP_DIR, WHICH),
		    map_location={'cuda:0': 'cuda:{}'.format(GPU_ID),
		                  'cuda:1': 'cuda:{}'.format(GPU_ID),
		                  'cuda:2': 'cuda:{}'.format(GPU_ID),
		                  'cuda:3': 'cuda:{}'.format(GPU_ID)})
		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for key, val in state_dict.items():
			name = key[7:]  # remove 'module'
			new_state_dict[name] = val
		model.load_state_dict(new_state_dict)

	model.cuda(GPU_ID)
	model.eval()
	print('Done.')

	"""
	Test
	"""
	print('Getting image ID...')

	gallery_cam, gallery_label = get_id(datasets['gallery'].fids)
	query_cam, query_label = get_id(datasets['query'].fids)
	print('Done.')

	# Extract feature
	print('Extracting gallery feature...')
	gallery_feature = extract_feature(model, dataloaders['gallery'], net_type=NET_TYPE,
		normalize_feature=normalize_feature, gpu=GPU_ID)
	print('Done.')
	print('Extracting query feature...')
	query_feature = extract_feature(model, dataloaders['query'], net_type=NET_TYPE,
		normalize_feature=normalize_feature, gpu=GPU_ID)
	print('Done.')

	# print(query_feature.size())

	# result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
	#           'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}

	# Save to Matlab for check	
	# scipy.io.savemat('pytorch_result.mat', result)
	
	query_cam = np.array(query_cam)
	query_label = np.array(query_label)
	gallery_cam = np.array(gallery_cam)
	gallery_label = np.array(gallery_label)

	######################################################################
	#print(query_feature.shape)
	print('Evaluating...')
	CMC = torch.IntTensor(len(gallery_label)).zero_()
	ap = 0.0
	#print(query_label)
	for i in range(len(query_label)):
	    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
	    if CMC_tmp[0] == -1:
	        continue
	    CMC = CMC + CMC_tmp
	    ap += ap_tmp
	    #print(i, CMC_tmp[0])

	CMC = CMC.float()
	CMC = CMC / len(query_label)  # average CMC
	print('Done.')
	print('top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))


if __name__ == '__main__':
	main()
