export model_name=MobileNet
export weight_file=./NIMA/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5
export image_source=./data/EVA_dataset/image

python ./NIMA/src/predict.py -b $model_name \
-w $weight_file \
-is $image_source \
-pf ./data/EVA_dataset/NIMA_config/pred.json \
-rf ./data/EVA_dataset/NIMA_config/samples_train.json