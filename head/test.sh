#srun -p VIFrontEnd --gres=gpu:8 python3 -u train_segmentation.py
srun -p VIFrontEnd --gres=gpu:1 python3 -u MBA_KL.py
