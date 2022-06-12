export model_name=MobileNet
export weight_file=./NIMA/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5
export image_source=C:/ML/GSOC/eva-dataset/images/EVA_together

python ./NIMA/src/predict.py -b $model_name -w $weight_file -is $image_source -pf ./data/pred.json