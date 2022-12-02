python train.py --outdir=~/training-runs --cfg=ffhq --data=/mnt/localssd/eg3d/dataset_preprocessing/ffhq/FFHQ_512.zip \
  --resume=/home/chuongh/eg3d/eg3d/networks/ffhqrebalanced512-128.pkl \
  --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True --neural_rendering_resolution_final=128 --outdir=/sensei-fs/users/chuongh/checkpoints/eg3d
