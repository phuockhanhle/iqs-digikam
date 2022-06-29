export job_dir=./data/AVA_dataset/NIMA_config/
export image_train_dir=D:/AVA_dataset/images/images
export image_test_dir=D:/AVA_dataset/images/images
#export image_train_dir=./data/EVA_dataset/image
#export image_test_dir=./data/EVA_dataset/image


python ./NIMA/src/train.py -j $job_dir -i $image_train_dir \
-is $image_test_dir \
-pf ./data/AVA_dataset/NIMA_config/pred.json \
-rf ./data/AVA_dataset/NIMA_config/samples_test.json \
#-ew ./NIMA/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5
