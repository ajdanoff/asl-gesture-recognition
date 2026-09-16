.PHONY: prepare-data train infer visualize

prepare-data:
	python scripts/convert_asl_to_yolo_mp.py

train:
	python scripts/train_asl_yolo_mp.py

infer:
	python scripts/infer_asl_yolo_mp.py

visualize:
	python scripts/visualize_asl_samples.py