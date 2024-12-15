# Fraud_Transaction_Detector
A Binary Classification Problem For Detection of Fraudulent Transaction in Bank. As a Solution RandomForestClassifier model is trained and tuned to attain  99% accuracy and 89% recall.

# Fraudulent Transaction Detection

## Overview
This project focuses on detecting fraudulent transactions using machine learning technique. The notebook explores data preprocessing, feature engineering, model training, and evaluation.

## Features
- **Data Preprocessing**: Handles missing values, encodes categorical data,scales features and undersampling of unbalanced class.
- **Model Building**: Implements machine learning algorithm to classify transactions as fraudulent or non-fraudulent- GPU accelerated RandomForestClassifier of CuML library is used to be 
                      trained on large dataset.
- **Model Fine-Tuning**:The Classifier is Fine tuned usng GridSearchCV and Custom tuning techniques for better performance.
- **Evaluation Metrics**: Provides insights into model performance using metrics such as accuracy, precision, recall, and F1 score.

## Dataset
- The dataset is the CSV file with 6362620 rows × 9 columns structure
- The Dataset link is [HERE](https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0)
- The Undersampled Dataset link is [HERE](https://drive.google.com/file/d/1BDRPic7J5t7jCSrX8s9HN-gdNxS16CPv/view?usp=sharing)
  
## Repository Structure
### `Data Dictionary.txt`: Contains info. of different features of dataset.
### `FraudTransacDetection.ipynb`: Well organized and Documented Notebook of complete workflow.
### `Task Details.pdf`: Company's guidelines on task.
### `gs_randomF_1.pkl`: Fine-tuned saved model using GridSearchCV.
### `seed_randomF_1_new.joblib`:***Final Fine-tuned saved model*** using Custom tuning by random seed generation (99% accuracy , 89% recall)

## Results
Summarize key findings and model performance here. For example:
- **Accuracy**: 99.78%
- **Precision**: 97.82%
- **Recall**: 88.86%
- **Recall**: 93.12%

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

## Author
Ashutosh Anand
