## Problem statement
Implement binary classification models for classifying Breast Cancer dataset as Benign vs Malignant. This Project includes
- Python Notebook which classifies the problem statement using 6 models and saves the related model artifacts
- Evaluates the result corresponding to each model
- Also include Streamlit App for Model testing and accuracy.

## Dataset Description
This dataset contains the Diagnostic Wisconsin Breast Cancer Database. Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing characteristics of cell nuclei present in the image. Sample images are available at http://www.cs.wisc.edu/~street/images/

dataset Fields:
- mean radius
- mean texture
- mean perimeter
- mean area
- mean smoothness
- mean compactness
- mean concavity
- mean concave points
- mean symmetry
- mean fractal dimension
- radius error
- texture error
- perimeter error
- area error',
- smoothness error
- compactness error
- concavity error
- concave points error
- symmetry error
- fractal dimension error
- worst radius
- worst texture
- worst perimeter
- worst area
- worst smoothness
- worst compactness
- worst concavity
- worst concave points
- worst symmetry
- worst fractal dimension
- target  - output label of the dataset

value count for output
- 1:  357
- 0:  212

## Models Used Evaluation

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| Decision Tree | 0.9474 | 0.9358 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| kNN | 0.9474 | 0.9817 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| Naive Bayes | 0.9649 | 0.9974 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| Random Forest (Ensemble) | 0.9649 | 0.9977 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| XGBoost (Ensemble) | 0.9561 | 0.9928 | 0.9583 | 0.9718 | 0.9650 | 0.9064 |

## Model Performance Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Logistic Regression remains most honest model which is performing well on all 4 quadrants for this simple linearly separable data as shown in the notebook as well. I think for this dataset Logistic regression is the best. |
| Decision Tree | since it has lowest AUC and MCC, then it might be an indicator that because of greedy nature of decision trees, data might be overfitting, we can pre prune it to have only depth of 3 if possible but overall performance is good. |
| kNN | Tied with Decision Tree on Accuracy (0.9474) and MCC (0.8880), but has a significantly higher AUC (0.9817). The high AUC indicates that kNN is actually very good at ranking the probability of a tumor being malignant, even if the final hard-label "Accuracy" is slightly lower than the top models. This suggests that with a bit of "K" value tuning (Hyperparameter optimization), it could match the top performers. |
| Naive Bayes | Performs Very strong! It achieved an MCC of 0.9253 and a high AUC of 0.9974 . It is a very stable, low-variance classifier. It performed better than the Decision Tree and kNN, proving that sometimes "simpler is better." |
| Random Forest (Ensemble) | Random Forest is perming well for all metrices except MCC, which means still LR is performing better then Random forest for this simple dataset|
| XGBoost (Ensemble) | XGBoost is mostly winner in all situation, I think dataset size is quite small and might be getting overoptimized for the same reason. However it is nearby LR, so I didn't tune it further|
