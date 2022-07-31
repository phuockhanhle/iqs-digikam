export model_name=InceptionResNetV2
export weight_file=C:/ML/iqs-digikam/data/combined_dataset/NIMA_config/weights/weights_inceptionresnetv2_06_0.895.hdf5
export image_source=D:/AVA_dataset/images/images
#export image_source=./data/EVA_dataset/image

python ./NIMA/src/predict.py -b $model_name \
-w $weight_file \
-is $image_source \
-pf ./data/combined_dataset/NIMA_config/pred.json \
-rf ./data/combined_dataset/NIMA_config/samples_test.json