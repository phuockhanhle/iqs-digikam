export job_dir=./data/EVA_dataset/NIMA_config/
export image_train_dir=./data/EVA_dataset/image
export image_test_dir=./data/EVA_dataset/image

python ./NIMA/src/train.py -j $job_dir -i $image_train_dir \
-is $image_test_dir \
-pf ./data/EVA_dataset/NIMA_config/pred.json \
-rf ./data/EVA_dataset/NIMA_config/samples_test.json