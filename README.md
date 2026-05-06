# Raisin-Classification-using-Na-ve-Bayes-and-Exploratory-Data-Analysis
This work aims to differentiate raisin types, Besni and Kecimen, by means of machine learning algorithms. The dataset involves the morphological aspects of raisin grains, and the primary objective is to clean, explore, and visualize the data and then use it to predict new cases employing a Gaussian Naïve Bayes model.
<img width="1100" height="734" alt="image" src="https://github.com/user-attachments/assets/19e6f45d-5f55-4f24-ab91-275f17af5445" />

By means of EDA, dimensionality reduction, and supervised learning, this paper presents a complete data mining pipeline that leads to a classification based on numerical descriptors in a highly effective manner.

The Naive Bayes classification algorithm is a probabilistic classifier. It is based on probability models that incorporate strong independence assumptions. The Naive Bayes model records how often a target field value appears together with a value of an input field.

The Naive Bayes classification algorithm includes the probability-threshold parameter ZeroProba. The value of the probability-threshold parameter is used if one of the above-mentioned dimensions of the cube is empty. A dimension is empty, if a training-data record with the combination of input-field value and target value does not exist.
Gaussian Naïve Bayes is the extension of naïve Bayes. 

While other functions are used to estimate data distribution, the Gaussian or normal distribution is the simplest to implement, as you will need to calculate the mean and standard deviation for the training data.

In this project, we create a Gaussian Naïve Bayes Model for classification.

About Data

DATASET 📑: https://www.muratkoklu.com/datasets/

Article Download (PDF): https://dergipark.org.tr/tr/download/article-file/1227592

In the study done by CINAR I., KOKLU M. and TASDEMIR S. (2020), a machine vision system was developed to distinguish between two different varieties of raisins (Kecimen and Besni) grown in Turkey. Firstly, a total of 900 pieces of raisin grains were obtained from an equal number of both varieties. These images were subjected to various preprocessing steps, and 7 morphological feature extraction operations were performed using image processing techniques. In addition, minimum, mean, maximum and standard deviation statistical information was calculated for each feature. The distributions of both raisin varieties on the features were examined, and these distributions were shown on the graphs. Later, models were created using LR, MLP, and SVM machine learning techniques and performance measurements were performed. The classification achieved 85.22% with LR, 86.33% with MLP and 86.44% with the highest classification accuracy obtained in the study with SVM. Considering the amount of data available, it is possible to say that the study was successful.

Dataset Columns

1.	Area
2.	MajorAxisLength
3.	MinorAxisLength
4.	Eccentricity
5.	ConvexArea
6.	Extent
7.	Perimeter
8.	Class

Libraries Used:

1.	pandas
2.	seaborn
3.	matplotlib
4.	scikit-learn

2. Dataset Description

The data was fetched from an Excel file listing 900 instances and several numerical attributes to describe raisin features. 
Class, the target variable, had two categories:
•	Besni
•	Kecimen

At first glance, the following conditions were met:

•	There were no missing values
•	There were no duplicates
•	All the features were numerical except the class label
•	The descriptive statistics gave a lot of useful information about distribution, central tendency, and spread.
3. Exploratory Data Analysis (EDA)

3.1 Pairplot Visualization

A pairplot was plotted to analyze the interaction among features, and it was also colored by class labels. This facilitated locating the possible separation of two groups of raisin samples at the very first glance.

3.2 Correlation Heatmap

The correlation matrix was used to show the relations between the numerical features. Quite a few attributes had correlations from moderate to strong, which gave the users guidance for feature interpretation and model behavior.

3.3 PCA Scatter Plot

PCA was used here to convert the data into two components. The scatter plot of these new features clearly demonstrated that the classes were clustered separately, thus confirming the data could be used for classification purposes.

4. Outlier Detection and Removal

Outlier removal was performed using the IQR method on all numerical features. The operation had gone from:

•	Original size: 900
•	After cleaning: 776

The boxplot provided visual support for verifying the outlier distribution.

By doing this, the model got trained on the dataset, which was more representative, thus less noisy and with better classification performance.

5. Data Preprocessing

5.1 Feature–Target Split

The data were separated into:

•	X: Numerical features
•	y: Class label

5.2 Train–Test Split

The data were split into train and test sets using an 80:20 ratio with stratification:

•	Train data: 620 records
•	Test data: 156 records

Stratification ensured that both the training and test sets had the same proportion of ‍‌‍‍‌‍‌‍‍‌classes.

6.‍‌‍‍‌‍‌‍‍‌ Model Development: Gaussian Naïve Bayes

The GaussianNB classifier was chosen as it is a very simple model, fast, and it has shown strong performance with normally distributed numeric data.

The model used the clean training set for fitting.

7. Model Evaluation

7.1 Predictions and Classification Report

Here is the classification report of the model:

•	Metric - Besni Kecimen
•	Precision - 0.85 0.82
•	Recall - 0.78 0.88
•	F1-Score - 0.82 0.85

Overall metrics:
•	Accuracy: 83.33%
•	Macro Average F1: 0.83
•	Weighted Average F1: 0.83
These metrics show that the model was able to discriminate between the two classes in a balanced and fair manner.

7.2 Confusion Matrix

The heatmap presentation of the confusion matrix confirmed:

•	That there were many correct classifications
•	Only very few instances were misclassified in both categories

7.3 Train-Test Accuracy Comparison

•	Training Accuracy: 81.77%
•	Testing Accuracy: 83.33%

Since training and testing accuracies are very close, it implies that:

•	There is no overfitting problem
•	The classifier generalizes well

8. Conclusion

This project presented a successful application of data mining and machine learning techniques to categorize raisin types through Naïve Bayes. By performing thorough EDA, removing outliers, reducing dimensionality, and evaluating performance, the pipeline showed strong predictive reliability.

The model accomplished:

•	83.33% accuracy on the unseen data
•	Good performance balance between the two raisin classes
•	Obvious separability in the PCA-reduced space

Future Improvements

•	To have better predictive results, one can attempt the following steps:
•	Trial with SVM, Random Forest, and Logistic Regression
•	Implementation of feature scaling or normalization
•	Using k-fold cross-validation for more reliable evaluation
•	Using GridSearchCV for hyperparameter tuning
•	With further ‍‌‍‍‌‍‌‍‍‌fine-tuning
