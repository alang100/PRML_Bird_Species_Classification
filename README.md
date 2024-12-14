# Bird Species Image Classification
This was my major Pattern Recognition and Machine Learning (PRML) Project, focusing on image classification using various machine learning techniques. As one of my first experiences in machine learning for image modelling, I invested significant effort into testing different approaches to image processing and data preparation. These included standardization, cross-validation with hyperparameter tuning, feature reduction using PCA, and other preprocessing techniques to optimize model performance. This allowed for a thorough exploration of how various methods impacted the accuracy and efficiency of different machine learning models.

This project evaluated multiple machine learning classification models, including Random Forest, SVM, and CNN, and compared their performance on a dataset of 10 bird species. The best results were achieved with a CNN model, which outperformed the other approaches with an accuracy of 83% on the test set.

Refer to the project report for a detailed description and the Jupyter Notebook file for details of the code, methodology and results:  

- [Project Report: PRML Bird Species Classification Final Report](PRML%20Bird%20Species%20Classififcation%20Final%20Report.pdf)
- [Jupyter Notebook Code: PRML Bird Species Classification Final Code](PRML%20Bird%20Species%20Classififcation%20Final%20Code.ipynb)
- [HTML Version: PRML Bird Species Classification Final Code](PRML%20Bird%20Species%20Classififcation%20Final%20Code.html)

---

### Project Abstract

The accurate identification of species is important for all forms of biological, ecological and evolutionary research. Many research projects require accurate identification of species including population monitoring, studying biodiversity of environmental habitats and the impact of climate change on species distribution. There is an abundance of images online today of birds that can be used in datasets to train machine learning algorithms to classify the species. 

This project developed various models utilising three machine learning algorithms, Random Forest, SVM and CNN to classify the images of 10 species of birds. Various methods were applied to the models and the dataset including standardization, cross-validation with hyperparameter tuning in a grid search and feature reduction using PCA (Principal Component Analysis). Overfitting was present in the training of all models. The best results achieved were with a CNN model with an accuracy of 83% on the test set. The CNN model proved to be more accurate than the best random forest (64%) and the SVM (69%). The dataset will be increased in future work and the models will be developed further.

---

### Dataset
The project utilized a subset of the “BIRDS 400 - SPECIES IMAGE CLASSIFICATION” dataset from Kaggle, which originally contains 400 bird species with 58,388 training images, 2,000 test images, and 2,000 validation images. Due to processing constraints, the dataset was reduced to 10 bird species, resulting in 1,475 training images (mean of 147.5 per species, ranging from 131 to 170).

Each image was resized to 224x224 pixels (150,528 total resolution), ensuring that only one bird occupied at least 50% of the image area. The dataset was curated to exclude species with significant male and female plumage differences and corrected for misclassified images, with 3.7% of the total removed. Additional images were sourced, cropped, and resized to maintain consistency and a minimum of 130 images per species.

### Methodology

The methodology for this project involved evaluating machine learning models, training classifiers, and testing the most effective model on unseen data:

1. **Evaluation of Classifiers**:  
   - Five common machine learning models were tested on a subset of the data.  
   - Support Vector Machine (SVM), Random Forest, and Logistic Regression were identified as the best-performing models.  
   - A Convolutional Neural Network (CNN) was included for its suitability in image classification tasks.

2. **Building and Training Models**:  
   - Data preprocessing steps included validating labels, splitting the data into training and test sets, and reshaping images for compatibility with the models.  
   - The models were trained using techniques such as:  
     - **Random Forest**: Tested with default settings, hyperparameter tuning using grid search, and increased estimators.  
     - **SVM**: Applied standardization, grid search for hyperparameter tuning, and Principal Component Analysis (PCA) for dimensionality reduction.  
     - **CNN**: Built with an 8-layer architecture, using techniques like convolutional and pooling layers, ReLU activation, and early stopping to reduce overfitting.

3. **Final Model Evaluation**:  
   - The most accurate model was evaluated on a validation set that was not used during training to ensure independent and reliable results.  

### Results and Evaluation Summary

This project evaluated over 11 models across three classifiers: Random Forest, SVM, and CNN, applying various configurations to enhance performance. Each successive model incorporated additional procedures such as standardization, Principal Component Analysis (PCA), and hyperparameter tuning using grid search.

- **Random Forest**: Accuracy ranged from 63.5% to 64.2%. Hyperparameter tuning with grid search and increasing the number of estimators did not significantly improve performance due to processing constraints.  
- **SVM**: Accuracy improved from 66.3% (default settings) to 68.8% with standardization, PCA (reducing image dimensions from 150,528 to 1,026), and hyperparameter tuning via grid search. PCA retained 99.5% of the variance.  
- **CNN**: The CNN model demonstrated superior accuracy, achieving 83% on the test set and 80% on an unseen validation set. Early stopping reduced overfitting and improved model stability.  

Overfitting was observed in all classifiers, with training accuracy consistently above 90%, but lower performance on the test set. Early stopping in the CNN mitigated this issue, with model loss minimized after six epochs. The results confirmed the CNN’s effectiveness for independent predictions, outperforming Random Forest and SVM significantly.


### Conclusion

Two configurations have been tried on the random forest model, five configurations have been applied to the support vector machine model and its dataset and two on the CNN.

The best model is clearly the CNN models. A more accurate result was obtained with a train-test ratio of 80/20.

The best SVM model which happens to be the last one evaluated has an overall accuracy of 68.2% at predicting the test data. It was found that to achieve the optimum performance of the model, the dataset had to be standardized so that all features have the same magnitude and standard deviation. Tuning the hyperparameters improved the performance of the default model by about 3% on the test dataset. PCA reduced the number of features from from 150528 to 1026 without any significant loss in accuracy. This will result in a reduction in processing time and complexity of the model.

The disadvantage of the SVM model is that when conducting the grid search with several hyperparameter combinations, it can take a lot longer for the grid search to determine the best model. Although in this dataset, tuning the hyperparameters did not yield any improvement in the model's accuracy, it will result in improvements for tuning similar models. Standardizing the dataset will result in speeding up the processing time by a factor of three which is very significant for large datasets.

Time permitted for some variations in the CNN models. The best model had a linear stack of 8 layers including conv 2d, max pooling, flatten and dense with a total of 44,397,002 trainable parameters. Early stopping prevented it from additional overfitting. This model achieved an overall accuracy score of 81.4% on the test set and 80.0% on the unseen validation set. This is a large improvement of accuracy by 15% over the best SVC and 20% over the best RFC.

Another important feature of a model's performance is its tradeoff between bias and variance. Bias can be defined as the difference between the average prediction of a model and the actual correct value. A model with high bias is underfit and will suffer from high inaccuracy in both the training and test data. Variance can be defined as the variability in a model's prediction for certain values in the training set. Models with high variance are overfit and will predict accurately on the training set but predict poorly on the test set. Models with high bias have low variance and vice versa. A good model should model the data well and have low bias and low variance.

In all models, high variance or overfitting was observed with cross-validation and in the test set. With progrssive models, the overfitting was reduced but it was still significant. More work will be done to address this issue on this dataset. In particular, the dataset needs to be oncreased significantly, preferably with additional images in the longer term or with data augmentation in the shorter term.

To summarise, the best model is the 8 layered CNN model with an 80/20 train-test split ratio.
