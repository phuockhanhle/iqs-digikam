# iqs-digikam

## Dataset : EVA and AVA dataset
EVA dataset :
- number samples : 8136
- mean score : 6.16 variance 1.09

AVA dataset :
- number samples : 39851
- mean score : 5.4504 variance 0.75137

Current problem : 
- data unbalance as most of sample's score is around 5. So, the model converges to predict image to 5. It will make the MSE -> variance of score. Hence, we can not conclude the performance of model
- training NIMA after a certain number of epoch would come to local minimal -> this problem is caused by imbalance of data

Ideal to improve :
- Get the range where most of score are in, consider this range as 0 -> 10 |> This doesn't change the fact that this data is in the middle, not in bad or good condition 
- Augment data of different range / reduce data of concentrated range in order to have balance dataset |> The synthesis image will follow the original image, so the model will not be generalized 
- Change metric that consider more result different region score 
  - consider the problem as classification and using F1 score to evaluate -> need to maximize F1 score instead of minimizing MSE 
  - Change the last dense of network to have regression problem


Principal idea is to combine different dataset to get as most generalized dataset as possible:
- problem :
  - how to have common sens label cross dataset -> how to set threshold to separate label 
  - Label of NIMA is list a from 0 -> 9, how to express this type of label
    - this kind of label is used to represent the distribution of human idea on an image, hence, reduce the effect of imbalance data
    - If we can combine various dataset, the problem of imbalance data would be resout, so, dont need to follow NIMA label format