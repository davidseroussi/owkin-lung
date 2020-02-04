# Owkin Challenge: Predicting lung cancer survival time

This is David Seroussi's participation at the Owkin Challenge.  
In order to run the notebooks, please place the train and test repositories at the root of the project.

# Summary

Having no experience in Survival analysis/Time-to-event analysis, I had to understand why these problems are different from typical supervised learning problems, and the potential methods that were available to address them. I also saw the provided data was quite limited, and the presence of censored data made the task more difficult.

This report is a summary of my work on this challenge.

The section ordering of this report is only partially chronological, considering I went back and forth between all the different parts throughout my work.


## Table of contents

1.  Data Exploration and Baseline model
    
2.  Research on previous works and ideas
    
3.  Feature Selection and Predictions
    
4.  Results
    

## 1) Data Exploration and Baseline model

In order to understand the problem better, I familiarized myself with the dataset and took a look at the distributions and correlations of both radiomics and clinical data. I also visualized the scans to have a concrete representation of what I was analyzing.

### a) Images
I had ideas about what could be done with the raw scans and their well segmented masks, but I also realized the challenges of having volumetric images; the only CNNs I had seen so far were for “flat” images. Because tumors are in 3D, information is contained in 3 directions which would be hard to extract with traditional CNNs.

### b) Radiomics
I found out that most radiomics features were not normally distributed, and were highly correlated between each other. This told me I would need to do some aggressive feature selection/engineering in order to get good generalized results, and not overfit the dataset.

### c) Clinical

The age of patients seemed quite normally distributed but had a couple of NAs. I thought we might me be able to replace them with the patients mean or median age in order not to lose any information as the dataset is already pretty small. After documenting myself about the TNM stage, I thought it could be a very good indicator combined with the age of the patients. However, in contrary to the T stage, the N stage and the M stage had a pretty uneven distribution, and would probably harm a model’s generalization.

  

### d) Baseline method

To make sure I was dealing properly with the data and to construct a “pipeline”, I reproduced the baseline model using the same features that were described in the challenge page.

  

## 2) Research on previous work and ideas

After exploring the dataset and having a better grasp the problem, I tried to find previous work on similar problems, especially feature selection in Survival analysis, Survival analysis using Neural networks and 3D CNNs.

The most helpful papers I found were:

- [A comparative study of machine learning methods for time-to-event survival data for radiomics risk modelling](https://www.nature.com/articles/s41598-017-13448-3)

This paper was the one that helped me achieve my best results. It reviews multiple ML methods for feature selection on radiomics data for Survival Analysis.

- [Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge](https://arxiv.org/abs/1802.10508)

This paper uses a modified 3DUNet in order to segment brain tumors, and calculated radiomics from the segmentation in order to do Survival analysis. Their available dataset is of the same magnitude as the lung dataset we have.


- [Time-to-Event Prediction with Neural Networks and Cox Regression](https://arxiv.org/abs/1907.00825)

This paper explains how the Cox Regression can be used as a loss for Neural Networks.
  
   
After reading these papers and exploring the dataset, here are the ideas I came up with:

-   Use Spearman Correlation to group radiomics features that are highly correlated between each other, eventually grouping them by type first (shape, glcm, firstorder, glrlm).
    
-   Use univariate and multivariate Cox Regression to find out most significant variables to use.
    
-   Train a 3DUnet on the images to predict the corresponding masks. As the UNet is basically an autoencoder, the feature vectors at the middle of the the network must contain important features about the tumor that might not be present in the radiomics. We could use these features as additional data, which was not done in the Brain Tumor paper (they only used the extracted radiomics).
- Collect more data, use the 3DUnet to segment more CT-scans and extract radiomics from these segmentations.
    

- We could also use a the Cox Regression loss for Neural Networks combined with UNet, but the training for 1D features is already very long, so a training with 3D convolutions would take an enormous amount of time. It could also overfit pretty easily since the dataset is relatively small.

## 3) Feature Selection and predictions
The goal here was to select/engineer features to improve the performance using the baseline model (Cox Regression) and parametric models like the Weibull AFT model.
I figured out that, because the C-index and the Log likelihood ratio should give similar results, the model was overfitting when these two metrics were highly different.

### a) Traditional methods
My first approach was to try traditional feature selection methods like Lasso, Tree-based models and Chi-squared. These methods all gave different features and the results using the Cox Regression were poorer than the baseline.
Because the radiomics are highly correlated, I did a Principal Component Analysis to construct new decorrelated features  and reduce the amount of dimension. By selecting the first two Principal Components which contained most of the variance, the performance slightly increased.
I also tried to use the risk predicted from the Cox Regression as an input to a Gradient Boosting algorithm, which again only slightly improved the performance.     

### b) Spearman Correlation
After trying the classical methods, I focused on the ones described in the [feature selection paper](https://www.nature.com/articles/s41598-017-13448-3).   
The main suggested methods were:  
- Average features having a high Spearman Correlation. Spearman Correlation benchmarks monotonic relationship whereas Pearson Correlation benchmarks linear relationships. Since the radiomics are extracted from images which are by nature non-linear, the Spearman Correlation might be better in this case.
- Select the best features from each type (shape, glcm, firstorder, glrlm).

After exploring these techniques by hand, I pseudo-automated the process, which is roughly this one:

1. Normalize features
2. Select significant features based on p-value < 0.005 using univariate/multivariate Cox Regression. The results greatly improved when, instead of doing an univariate regression on each of the features, I added the Source Dataset as an input. This gave features that are significant over the two datasets.
4. Average features having Spearman Correlation > 0.8 (or 0.9)
5. Cross-validate on different models
6. Repeat steps 2-3-4 until good C-index and Log likelihood ratio

I did not fully automate this pipeline in order to keep control on which features were being used.    
Sometimes I added previous significant features that were discarded in earlier steps; for example if I saw that shape features were not used at all in the grouped features I added the most significant shape feature.  
I also tried to split features by type before averaging them, which led to more overfitting.

I studied the significance of the clinical data using the same principles, and found out that the SourceDataset, the age and the Tstage are the most sginificant features.

This method gave the best results on the training and test set.  
All models tested (Cox Regression, Weibull AFT model, log-normal and log-logistic AFT models) gave similar results.

### c) Other approaches

These are approaches I explored but did not invest too much time in, as I thought feature selection was the key to this challenge.

- I implemented my "idea" about extracting feature vectors of tumors from the images using a 3DUNet (see this [colab notebook](3DUnet_Survival.ipynb)), and used these features as input data, which did not work very well. I did not spend much time optimizing it, but it might a good solution since neural networks are good at extracting abstract features that are not obvious to humans.

- I also tried [DeepSurv](https://github.com/liupei101/TFDeepSurv) with the radiomics data, but the training was astonishly long and the results improved very very slowly. 

This list is not exhaustive as I tried many different approaches and sometimes combined them.

## 4) Results
These are the best results I obtained, both on the training set and the public test set.  

Using the method described in **3.b)**, I selected 5 features, with 2 of them being groups of averaged radiomics.  
The features are :
- 'group2': average of 'original_firstorder_Mean', 'original_firstorder_Median', 'original_glcm_Id', 'original_glcm_Idm','original_glcm_Idn'
- 'group4': original_glcm_ClusterProminence
- 'SourceDataset'
- 'Nstage'
- 'age'

The model used is a CoxPH model with the [lifelines](https://lifelines.readthedocs.io/en/latest/) library.

Results on train set:  
C-Index: 0.71 (+/- 0.06)  
Log-likelihood ratio test: 87.04
Max: 0.77  
Min: 0.63  

Results on pulic test set:  
C-Index: 0.7314  

Getting a quite high Concordance Index on the public test set, and because the model has some variance, I expect the performance on the private test to be lower (C-Index ~= 0.69) than on the public set.
