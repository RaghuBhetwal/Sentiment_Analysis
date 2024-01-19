# Sentiment Analysis Project

## Introduction
This project focuses on sentiment analysis of Comments using machine learning techniques. The goal is to predict the sentiment of textual comments using natural language processing (NLP) methods.

## Data
The project uses a dataset (`data.csv`) containing comments and their corresponding sentiment labels.YoutubeAPI is being used to extract data from "LORD OF RINGS" trailler.
The distribution of sentiment classes is explored, and data preprocessing steps are applied.

## Methodology
The following steps are performed in the project:
- Data loading and exploration
- Text preprocessing, including lowercasing, punctuation removal, tokenization, stopword removal, lemmatization, and emoji removal
- Encoding sentiment labels to numerical values
- Splitting the dataset into training and testing sets
- Vectorizing processed comments using TF-IDF
- Training ML models
- Hyperparameter tuning using Grid Search
- Building a Voting Classifier with multiple base models
- Evaluating models and visualizing results
- BERT-based Sentiment Analysis
  
![image](https://github.com/RaghuBhetwal/Sentimenr_analysis/assets/88844603/04bbd3bb-c74a-46c9-a7c6-e3d6f3be756f)


## BERT-Based Sentiment Analysis
This script performs sentiment analysis on text data using BERT (Bidirectional Encoder Representations from Transformers). The sentiment analysis is based on a dataset stored in the file `data.csv`.


### Kaggle Environment Setup

This Python script is designed to run in a Kaggle environment. Kaggle environments come with many helpful analytics libraries installed. The environment is defined by the kaggle/python Docker image. For example, it includes packages like NumPy and pandas.
`The reason behind using kaggle is to provide its GPU resources`

## Requirements
The BERT project requires the following Python librarie :

- numpy==1.23.5
- pandas==1.4.4
- torch== 2.0.1
- seaborn==0.11.2
- matplotlib==3.5.2
- transformers==4.30.2
- scikit-learn==1.0.2
- nltk==3.7



```
 install and import libraries
```
example:
```
import pandas as pd

```
## Data Loading and Preprocessing
The script loads the dataset from the file data.csv and performs several preprocessing steps, including converting string labels to numerical values and tokenization using the BERT tokenizer.

## BERT Model Setup
The script uses the BERT model for sequence classification. It initializes the BERT model without loading the classifier weights and modifies the classifier for sentiment analysis.

## Training and Evaluation
The script implements a training loop with cross-validation, validation loss tracking, and early stopping. After training, the model is evaluated on the test set. The training and validation loss, accuracy, and confusion matrix are visualized.

BERT: Confusion Matrix:


![image-1](https://github.com/RaghuBhetwal/Sentimenr_analysis/assets/88844603/1360c866-fff4-417d-bbb0-455a04b060c9)


## Machine Learning model
### SVM Model
- A Support Vector Machine (SVM) model is trained using TF-IDF vectorized comments.
- The model is evaluated using accuracy and a classification report.

### Hyperparameter Tuning
- Grid Search is employed to find the best hyperparameters for the SVM model.
- The tuned model is evaluated, and results are presented.

### Voting Classifier
- A Voting Classifier is created with multiple base models: SVM, Logistic Regression, and Naive Bayes.
- Models are initially trained without hyperparameter tuning, and results are presented.
- Grid Search is used to find the best hyperparameters for the Voting Classifier.
- The tuned Voting Classifier is evaluated and results are presented.

## Results
### SVM Model
- Accuracy: 0.68
- Accuracy with HyperParameter Tuning: 0.71

### Voting Classifier
- Accuracy:  0.67
- Accuracy with HyperParameter Tuning:0.69 


## Confusion Matrix
Confusion matrices are visualized for the tuned SVM model and the Voting Classifier.

![image-2](https://github.com/RaghuBhetwal/Sentimenr_analysis/assets/88844603/55fdb84a-21fa-46a8-8c96-156f5c43b9cd)

![image-3](https://github.com/RaghuBhetwal/Sentimenr_analysis/assets/88844603/167124c3-448f-47c8-af17-5ec0d68a8c3d)



## Files and Directories
- **notebook.ipynb:** ``code/ML.ipynb``, ``code/bert.ipynb``
- **data/:** ``data/data.csv ``
- **models/:**
  - [svm_tuned_model.pkl: Trained SVM model with hyperparameter tuning]
  - [voting_classifier_without_tuning_model.pkl: Voting Classifier without hyperparameter tuning]
  - [voting_classifier_with_tuning_model.pkl: Tuned Voting Classifier with hyperparameter tuning]

## How to Run
1. Ensure you have Python installed.
2. Navigate to the project directory in the terminal.
3. Open the Jupyter Notebook (`ML.ipynb`).
4. Run the first cell in the notebook after installation of required libraries.
5. You can find `requirements.txt` file for libraries and also included here.

## Requirements
The project requires the following Python libraries, which will be installed by running the first cell in the Jupyter Notebook:

- pandas==1.4.4
- scikit-learn==1.0.2
- nltk== 3.7
- matplotlib==3.5.2
- seaborn==0.11.2
- joblib==1.3.2

use below command with above requirements
```
!pip install ........
```
```
run each cell one by one from notebook
```


## Acknowledgments

I would like to express my sincere gratitude to Professor for their invaluable guidance and support throughout this project. Their expertise and encouragement have played a significant role in shaping the direction of this work. I am also thankful for team Ms. Alexandra. This project has been an enriching learning experience under their mentorship.

## Conclusion

In conclusion, I am grateful for the opportunity to work on this sentiment analysis project. The insights gained and lessons learned will undoubtedly contribute to my growth in the field of machine learning and natural language processing.

Thank you to everyone who has been part of this journey.

## License
``MIT License ``
