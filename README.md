# Predicting County Presidential Winners

## Dependencies: OpenCV 2.4 and Python 2.7

## Model Selection:
I tested four models using a Linear SVM:
  1. Train on zero mean unit variance (zmuv) normalized features + PCA
  2. Train on zmuv features (no PCA)
  3. Train on PCA features (no zmuv)
  4. Train on regular features (no PCA, no zmuv)
Using the metric of error rate calculated by testing the training data, I chose the third model. Also, I believed uncorrelated features to be highly important to this problem.

## Closing remarks:
It may seem absurd that I chose to forgo feature normalization, but feature normalization isnt always a good thing.
Normalizing features assigns the same weights to every feature. In this case, not all features are created equal, apparently.
As to why I persisted with a Linear SVM and did not try any other models like a Fisher Linear Discriminant...
The main constraint was time, it always is. Given more time, this would be a great research project, but Im sure some smart person is already working on it.
Given more time, I would definitely look at some other linear classifiers and maybe some nonlinear ones. If I had a LOT of time Id look at neural nets!

Richard Liu, Nov 19 2015