export model_name=InceptionResNetV2
export weight_file=./data/combined_dataset/NIMA_config/weights/weights_inceptionresnetv2_11_1.612.hdf5
export image_source=D:/AVA_dataset/images/images
#export image_source=./data/EVA_dataset/image

python ./NIMA/src/predict.py -b $model_name \
-w $weight_file \
-is $image_source \
-pf ./data/combined_dataset/NIMA_config/pred.json \
-rf ./data/combined_dataset/NIMA_config/samples_test.json