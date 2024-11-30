# PRML_Bird_Species_Classification
This was my major Pattern Recognition and Machine Learning Project. It is a an image classification project using various machine learning techniques.

This was an evaluation of various machine learning classification models and data preparation steps to evaluate how they performed and compared to each other. 

Refer to the project report for a summary and the Jupyter Notebook file for details of the code and results.

---

### Project Abstract

The accurate identification of species is
important for all forms of biological, ecological and
evolutionary research. Many research projects
require accurate identification of species including:
population monitoring, studying biodiversity of
environmental habitats and the impact of climate
change on species distribution. There is an
abundance of images online today of birds that can
be used in datasets to train machine learning
algorithms to classify the species. This project
developed various models utilising three machine
learning algorithms, Random Forest, SVM and
CNN to classify the images of 10 species of birds.
Various methods were applied to the models and the
dataset including standardization, cross-validation
with hyperparameter tuning in a grid search and
feature reduction using PCA (Principal Component Analysis). Overfitting was
present in the training of all models. The best results
achieved were with a CNN model with an accuracy
of 83% on the test set. The CNN model proved to be
more accurate than the best random forest (64%)
and the SVM (69%). The dataset will be increased in
future work and the models will be developed
further.

---

### Conclusion

Two configurations have been tried on the random forest model, five configurations have been applied to the support vector machine model and its dataset and two on the CNN.

The best model is clearly the CNN models. A more accurate result was obtained with a train-test ratio of 80/20.

The best SVM model which happens to be the last one evaluated has an overall accuracy of 68.2% at predicting the test data. It was found that to achieve the optimum performance of the model, the dataset had to be standardized so that all features have the same magnitude and standard deviation. Tuning the hyperparameters improved the performance of the default model by about 3% on the test dataset. PCA reduced the number of features from from 150528 to 1026 without any significant loss in accuracy. This will result in a reduction in processing time and complexity of the model.

The disadvantage of the SVM model is that when conducting the grid search with several hyperparameter combinations, it can take a lot longer for the grid search to determine the best model. Although in this dataset, tuning the hyperparameters did not yield any improvement in the model's accuracy, it will result in improvements for tuning similar models. Standardizing the dataset will result in speeding up the processing time by a factor of three which is very significant for large datasets.

Time permitted for some variations in the CNN models. The best model had a linear stack of 8 layers including conv 2d, max pooling, flatten and dense with a total of 44,397,002 trainable parameters. Early stopping prevented it from additional overfitting. This model achieved an overall accuracy score of 81.4% on the test set and 80.0% on the unseen validation set. This is a large improvement of accuracy by 15% over the best SVC and 20% over the best RFC.

Another important feature of a model's performance is its tradeoff between bias and variance. Bias can be defined as the difference between the average prediction of a model and the actual correct value [13]. A model with high bias is underfit and will suffer from high inaccuracy in both the training and test data. Variance can be defined as the variability in a model's prediction for certain values in the training set. Models with high variance are overfit and will predict accurately on the training set but predict poorly on the test set. Models with high bias have low variance and vice versa. A good model should model the data well and have low bias and low variance.

In all models, high variance or overfitting was observed with cross-validation and in the test set. With progrssive models, the overfitting was reduced but it was still significant. More work will be done to address this issue on this dataset. In particular, the dataset needs to be oncreased significantly, preferably with additional images in the longer term or with data augmentation in the shorter term.

To summarise, the best model is the 8 layered CNN model with an 80/20 train-test split ratio.
