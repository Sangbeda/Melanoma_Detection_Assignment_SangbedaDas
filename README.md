# Melanoma Detection through multiclass classification using CNN
> Building a multiclass classification model using a custom convolutional neural network in TensorFlow, which can accurately detect melanoma, which is a type of skin cancer. 


## Table of Contents
* [General Info](#general-information)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)


## General Information
- Melanoma is a type of skin cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
- The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.
The data set contains the following diseases:
1. Actinic keratosis
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion
- Project Pipeline
1. Data Reading/Data Understanding → Defining the path for train and test images 
2. Dataset Creation→ Create train & validation dataset from the train directory with a batch size of 32. Also, make sure you resize your images to 180*180.
3. Dataset visualisation → Create a code to visualize one instance of all the nine classes present in the dataset 
4. Model Building & training : 
 - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
 - Choose an appropriate optimiser and loss function for model training
 - Train the model for ~20 epochs
 - Findings after the model fit: Any evidence of model overfit or underfit?
5. Chose an appropriate data augmentation strategy to resolve underfitting/overfitting 
6. Model Building & training on the augmented data :
 - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
 - Choose an appropriate optimiser and loss function for model training
 - Train the model for ~20 epochs
 - Findings after the model fit: is the earlier issue resolved?
7. Class distribution: Examine the current class distribution in the training dataset 
 - Which class has the least number of samples?
 - Which classes dominate the data in terms of the proportionate number of samples?
8. Handling class imbalances: Rectify class imbalances present in the training dataset with Augmentor library.
9. Model Building & training on the rectified class imbalance data :
 - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
 - Choose an appropriate optimiser and loss function for model training
 - Train the model for ~30 epochs
 - Findings after the model fit: are the issues resolved?


## Conclusions
- Initially, the model built on the raw train dataset with a batch size of 32 and epoch of 20 revealed the following:
 1. Loss and Accuracy Trends:
 - Training and Validation Loss: The training loss (measured by "loss") and validation loss (measured by "val_loss") are decreasing over the epochs, which is a positive sign. This indicates that the model is learning from the data and becoming better at making predictions.
 - Training and Validation Accuracy: The training accuracy (measured by "accuracy") and validation accuracy (measured by "val_accuracy") are increasing over the epochs. This suggests that the model is improving in its ability to classify the data correctly.
 - The validation loss and accuracy show a fluctuating pattern. It is important to monitor these metrics to check for overfitting.

 2. Overfitting:
 - There is a slight gap between the training and validation performance. This is more noticeable in the later epochs, where the training accuracy is higher than the validation accuracy. It's a sign that the model might be overfitting the training data.
 - To address overfitting, we could consider applying regularization techniques, adjusting model complexity, or increasing the size of the dataset. We could also consider data augmentation to increase the diversity of training data and potentially improve generalization.

- Upon applying augmentation to mitigate overfitting, a dramatic change in model performance is observed, which can be attributed to various factors:
 1. Data Augmentation Quality: The choice of data augmentation techniques and their parameters can greatly impact the quality of augmented data. In the post-augmentation scenario, it appears that augmentation may have introduced too much variability or distortion, leading to lower model performance. This requires further fine-tuning of the augmentation parameters to produce more realistic and helpful augmented images.
 2. Overfitting: The model's performance after augmentation is worse because of overfitting. While augmentation helps to reduce overfitting by providing more diverse training examples, the increased variability in augmented data might have made it harder for the model to generalize. To address this, regularization techniques such as dropout and L2 regularization could be leveraged to combat overfitting. Further tuning of the hyperparameters could also be considered.
 3. Class Imbalance: Data augmentation should ideally maintain class balance. Perhaps the original dataset was imbalanced and augmentation didn't address this properly, thereby affecting the model's performance. The augmentation process must maintain class distribution.

- Class distribution analysis revealed seborrheic keratosis to be the class with the least number of samples, whereas the classes which dominate the data in terms of the proportionate number of samples include melanoma (19.56%), pigmented benign keratosis (20.63%), nevus (15.94%), and basal cell carcinoma (16.79%).

- An analysis of the output and the model's performance on the rectified class imbalance data is provided below:

 1. Loss and Accuracy Trends:

 - The output shows the training and validation loss and accuracy for each epoch. Loss is a measure of how well the model is performing, with lower values indicating better performance. Accuracy indicates the proportion of correctly classified samples.
 - At the beginning of training (Epoch 1), both training and validation accuracy are relatively low. The model's accuracy is approximately 17.84% on the training data and 21.48% on the validation data.
 - As training progresses, both training and validation accuracy improve. By Epoch 4, training accuracy reaches around 48.03%, and validation accuracy is approximately 50.56%.
 - The model continues to improve, reaching an accuracy of around 66.20% on the training data and 61.07% on the validation data by the end of training (Epoch 30).

 2. Loss Trends:

 - Training loss is relatively high at the beginning but gradually decreases, indicating that the model is learning from the data. By the end of training, the training loss is 0.9601.
 - Validation loss shows a similar trend, decreasing over time. By the end of training, the validation loss is 1.1817.

 3. Overfitting:

 - Initially, the model's training accuracy is lower than the validation accuracy, suggesting underfitting. As training progresses, the training accuracy surpasses the validation accuracy, indicating overfitting.
 
 4. Training Time:

 - The training time per epoch is also provided. In this case, each epoch takes around 29 to 38 seconds.

 5. Model Performance:

 - The final validation accuracy of around 61.07% suggests that the model can correctly classify approximately 61% of the validation samples. The model has learned to some extent from the training data but may still benefit from further optimization or the application of advanced architectures or techniques.
 - To improve model performance, we could consider the following steps:
   - Experiment with different model architectures or hyperparameters.
   - Implement regularization techniques such as L2 regularization to mitigate overfitting.
   - Further adjust the learning rate and learning rate schedule.
   - Increase the training dataset size to expose the model to more variations in the data.
 It's important to monitor training and validation metrics closely and use these insights to fine-tune your model and achieve better results.


## Technologies Used
- Python - version 3.10.12
- Tensorflow - version 2.13.0
- NumPy - version 1.23.5
- Matplotlib - version 3.7.0
- Pandas - 1.5.3


## Contact
Created by Sangbeda Das (sangbeda.das@gmail.com) - feel free to contact me!
