# Tech4MentalHealth


---

# Mental Health Text Classification – RoBERTa Model

## Project Overview

This project implements a machine learning model to classify text statements written by university students in Kenya regarding mental health challenges.

The objective is to support the development of a mental health chatbot capable of detecting four categories:

* Depression
* Alcohol
* Suicide
* Drugs

The evaluation metric for the challenge is Log Loss, which requires probabilistic predictions for each class.

---

## Dataset Description

### Train.csv

| Column | Description                   |
| ------ | ----------------------------- |
| ID     | Unique identifier             |
| text   | Student statement or question |
| label  | Target class                  |

### Test.csv

| Column | Description                   |
| ------ | ----------------------------- |
| ID     | Unique identifier             |
| text   | Student statement or question |

### SampleSubmission.csv

| Column     | Description               |
| ---------- | ------------------------- |
| ID         | Unique identifier         |
| Depression | Probability of Depression |
| Alcohol    | Probability of Alcohol    |
| Suicide    | Probability of Suicide    |
| Drugs      | Probability of Drugs      |

---

## Model Architecture

This solution uses:

* RoBERTa-base transformer model
* Dropout regularization
* Fully connected classification layer with 4 outputs
* Softmax activation for probability outputs

### Architecture Flow

Text → Tokenizer → RoBERTa → CLS token representation → Dropout → Linear layer → Softmax

The output is a probability distribution over the four classes.

---

## Training Strategy

### Data Preprocessing

* Convert all text to lowercase
* Remove duplicate text-label pairs
* Encode labels as follows:

  * Depression → 0
  * Alcohol → 1
  * Suicide → 2
  * Drugs → 3

---

### Tokenization

* RobertaTokenizer is used
* Maximum sequence length: 64 tokens
* Padding and truncation applied

---

### Cross-Validation

* Stratified 5-Fold Cross Validation
* Preserves class distribution across folds
* Reduces overfitting
* Produces stable validation Log Loss

---

### Optimization

* Optimizer: AdamW
* Learning rate: 2e-5
* Scheduler: Linear learning rate scheduler with warmup
* Epochs: 3
* Batch size: 16

---

## Evaluation Metric

The validation metric is Log Loss (Cross-Entropy Loss).

Lower values indicate better performance.

The final test predictions are:

* Averaged across the 5 folds
* Converted to probabilities using Softmax
* Saved as submission.csv

---

## Output Format

The generated submission file follows this structure:

```csv
ID,Depression,Alcohol,Suicide,Drugs
02V56KMO,0.72,0.05,0.10,0.13
03BMGTOK,0.80,0.02,0.15,0.03
```

Each row contains probabilities that sum to 1.

---

## Requirements

Install required dependencies:

```bash
pip install transformers torch scikit-learn pandas numpy
```

A GPU with CUDA support is recommended but not mandatory.

---

## How to Run

1. Place the following files in the working directory:

   * Train.csv
   * Test.csv
   * SampleSubmission.csv

2. Run the notebook or Python script.

3. After training completes, submission.csv will be generated.

4. Upload submission.csv to the competition platform.

---

## Expected Performance

Typical cross-validation performance:

* CV Log Loss approximately between 0.35 and 0.45

Performance may vary depending on:

* Random seed
* Number of training epochs
* Hardware configuration

---

## Possible Improvements

Performance may be improved by:

* Increasing the number of epochs
* Using roberta-large
* Applying label smoothing
* Adding text data augmentation
* Ensembling multiple random seeds
* Using repeated cross-validation

---

## Project Objective

This model serves as a prototype component of a mental health chatbot designed to:

* Improve accessibility to mental health support
* Assist early detection of mental health risk signals
* Support university students in Kenya

---

## Ethical Considerations

This model is intended for research and prototyping purposes only.

It must not replace professional medical diagnosis or certified mental health services.





````md
---

# Mental Health Text Classification – Linear SVC Model

## Overview

In addition to the RoBERTa-based deep learning model, this project also includes a traditional machine learning approach implemented in **model_svc.ipynb**.

This model uses:

* TF-IDF feature extraction
* Linear Support Vector Classification (LinearSVC)
* Probability calibration for Log Loss optimization

The SVC model serves as a strong baseline for comparison with transformer-based models.

---

## Model Pipeline

The Linear SVC system is implemented using a Scikit-learn pipeline with the following stages:

Text → TF-IDF Vectorizer → LinearSVC → Probability Calibration

### Components

* **TF-IDF Vectorizer**
  * Converts text into numerical feature vectors
  * Uses unigrams and bigrams
  * Removes English stopwords
  * Limits vocabulary size to reduce overfitting

* **LinearSVC**
  * Linear Support Vector Machine classifier
  * Optimized for high-dimensional sparse text data
  * Trained using hinge loss

* **CalibratedClassifierCV**
  * Wraps LinearSVC
  * Converts decision scores into probabilities
  * Required for Log Loss evaluation

---

## Data Preprocessing

Before training, the following preprocessing steps are applied:

* Convert text to lowercase
* Remove duplicate text-label pairs
* Drop missing values
* Shuffle the dataset
* Encode labels as integers

Example preprocessing:

```python
df["text"] = df["text"].str.lower()
df = df.drop_duplicates(subset=["text", "label"])
````

---

## Feature Extraction (TF-IDF)

TF-IDF is used to transform text into numerical vectors:

* Maximum features: 50,000
* N-grams: (1,2)
* Sublinear TF scaling
* L2 normalization

This representation captures both word importance and contextual patterns.

---

## Training Strategy

### Cross-Validation

* Stratified K-Fold (5 folds)
* Maintains class balance
* Reduces variance
* Improves generalization

### Calibration

Because LinearSVC does not output probabilities, calibration is applied:

* Method: Sigmoid
* Cross-validation inside calibration
* Produces reliable probability estimates

---

## Model Training

The model is trained using a Scikit-learn pipeline:

```python
Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svc", LinearSVC())
])
```

Then wrapped with:

```python
CalibratedClassifierCV(...)
```

This enables probability prediction.

---

## Evaluation

The Linear SVC model is evaluated using:

* Log Loss (primary metric)
* Accuracy
* Confusion Matrix
* Classification Report

Validation results are computed using cross-validation.

Lower Log Loss indicates better probabilistic predictions.

---

## Prediction and Submission

After training, the calibrated model predicts probabilities for the test set.

Output format:

```csv
ID,Depression,Alcohol,Suicide,Drugs
02V56KMO,0.45,0.20,0.25,0.10
```

Each row sums to 1.

The file is saved as:

```
submission_svc.csv
```

---

## Visualization

The notebook includes visualizations for:

* Class distribution
* Confusion matrix
* Decision margin projection (PCA)
* Support vector boundaries (2D approximation)

These visualizations help understand:

* Class separability
* Misclassification patterns
* Model confidence

---

## Advantages of Linear SVC

* Fast training
* Low memory usage
* Works well on small datasets
* Interpretable feature weights
* Strong baseline performance

---

## Limitations

* Cannot model deep semantic context
* Sensitive to noisy labels
* Requires manual feature engineering
* Lower performance than transformers on complex language

---

## Comparison with RoBERTa

| Feature                  | Linear SVC | RoBERTa |
| ------------------------ | ---------- | ------- |
| Training Time            | Fast       | Slow    |
| GPU Required             | No         | Yes     |
| Contextual Understanding | Low        | High    |
| Accuracy                 | Medium     | High    |
| Complexity               | Low        | High    |

Linear SVC is mainly used as a baseline and for rapid experimentation.

---

## How to Run (SVC Model)

1. Open `model_svc.ipynb`
2. Ensure Train.csv and Test.csv are present
3. Run all cells
4. Generated file: `submission_svc.csv`

---

## Expected Performance (SVC)

Typical results:

* Log Loss: 0.55 – 0.70
* Accuracy: 65% – 75%

Performance depends on:

* Feature size
* Regularization parameter
* Data quality

---

## Future Improvements

Possible enhancements for the SVC model:

* Hyperparameter tuning (GridSearchCV)
* Class-weight balancing
* Advanced preprocessing (lemmatization)
* Ensemble with neural models
* Better probability calibration

---

## Purpose of SVC Module

The SVC model provides:

* A lightweight baseline
* Fast prototyping
* Model interpretability
* Benchmark for deep learning models

It complements the RoBERTa system and supports comparative evaluation.

---

```
```
