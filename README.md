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
