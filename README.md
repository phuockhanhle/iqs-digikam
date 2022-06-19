# iqs-digikam

## Dataset : EVA and AVA dataset
EVA dataset :
- number samples : 8136
- mean score : 6.16 variance 1.09

AVA dataset :
- number samples : 39851
- mean score : 5.4504 variance 0.75137

Current problem : data unbalance as most of sample's score is around 5. So, the model converges to predict image to 5. It will make the MSE -> variance of score. Hence, we can not conclude the performance of model

Ideal to improve :
- Get the range where most of score are in, consider this range as 0 -> 10
- Augment data of different range / reduce data of concentrated range in order to have balance dataset
- Change metric that consider more result different region score
