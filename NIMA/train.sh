export job_dir=./data/combined_dataset/NIMA_config/
export image_train_dir=D:/AVA_dataset/images/images
export image_test_dir=D:/AVA_dataset/images/images
#export image_train_dir=./data/EVA_dataset/image
#export image_test_dir=./data/EVA_dataset/image


python ./NIMA/src/train.py -j $job_dir \
-it $image_train_dir \
-ie $image_test_dir \
-pf ./data/combined_dataset/NIMA_config/pred.json \
-rf ./data/combined_dataset/NIMA_config/samples_test.json \
#-ew ./data/AVA_dataset/NIMA_config/weights/weights_vgg16_11_0.907.hdf5
