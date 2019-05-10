train_posenet:
	python3 train.py --model posenet --dataroot ./datasets/KingsCollege --name posenet/KingsCollege/beta500 --beta 500 --gpu 0 --display_id 1

train_poselstm:
	python3 train.py --model poselstm --dataroot ./datasets/KingsCollege --name poselstm/KingsCollege/beta500 --beta 500 --niter 1200 --gpu 0 --display_id 1

train_cloudnet:
	python3 train.py  --model cloudnet --dataroot ./datasets/Carla/episode_001 --display_id 1

train_cloudnet_images:
	python3 train.py  --model cloudnet --dataroot ./datasets/Carla/episode_001 --input_type rgb --display_id 1
