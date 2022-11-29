python train.py --outdir=~/training-runs --cfg=ffhq --data=/vulcanscratch/chuonghm/FFHQ_512_1k.zip \
  --resume=networks/ffhq512-128.pkl \
  --gpus=1 --batch=1 --gamma=1 --gen_pose_cond=True --neural_rendering_resolution_final=128
