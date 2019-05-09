train_posenet:
	python3 train.py --model posenet --dataroot ./datasets/KingsCollege --name posenet/KingsCollege/beta500 --beta 500 --gpu 0 --display_id 1

train_poselstm:
	python3 train.py --model poselstm --dataroot ./datasets/KingsCollege --name poselstm/KingsCollege/beta500 --beta 500 --niter 1200 --gpu 0 --display_id 1

train_cloudnet:
	python3 train.py --dataroot 0 --model cloudnet --input_type point_cloud  --display_id 1
