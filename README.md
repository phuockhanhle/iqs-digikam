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

One of the main question is how to evaluate the generalisation of the model. Using one dataset makes us fall to the imbalance of that data. But combing various dataset causes the problem of balancing the properties of all dataset. At last, we can present the dataset on separated datasets, but using a good metric. Metric like MSE or F1 score will get the problem of scale between different dataset. Hence, I propose using Pearson Correlation score.

As the model will serve as a classification of 3 class, in digiKam, I decided to have a metric of 3 class classification [0 1 2] represent bad, normal and good image. The question is how to separate them reasonably.

In order to perform the score by the class of digikam, I set 2 threshold of score to separating the class. As the purpose is to have a balanced data, I choose the threshold is 5 and 6, it means if the score <= 5, the image is labeled as 0, 5 < score <= 6 is for normal, and the rest is for good image. **we need to notice that these thresholds is not choose by a good reference**.

The model also should be changed to adapt the new context, the last layer is changed to softmax with 3 classes.
The result is not so good, F1 score on each class is 0.62857742 0.43022272 0.58738075 and the accuracy is 0.5438715820976738. That means there are only a half of case it gives a right prediction.
The main reason is that the images that are labeled 0 is not really bad, so, it gives the confusion for model. 

The solution is to concate several dataset, and get bad image from different dataset, like koniq10k.