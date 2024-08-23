# Decision Tree
This repository contains a machine learning project that predicts the quality of red wine using a decision tree classifier. The model is trained on the **Red Wine Quality** dataset, which consists of physicochemical properties of red wine and their corresponding quality ratings.

## Dataset

The dataset used for this project is the **Wine Quality Data Set** from the UCI Machine Learning Repository.

- **Features**: The dataset contains the following features related to physicochemical properties of wine:
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
  
- **Target**: The target variable is the `quality` of the wine, which is a categorical rating on a scale from 3 to 8.

## Project Structure
- `winequality-red.csv`: The dataset file containing all the physicochemical properties and wine quality ratings.
- `red_wine_quality.py`: The Python script containing the full implementation of the decision tree classifier, confusion matrix, and classification report.

## Installation and Requirements
To run this project, you need to install the following Python packages:

```bash
pip install pandas scikit-learn matplotlib seaborn
```
## Running the Project
1) Clone the repository:
```bash
git clone https://github.com/himpar21/Decision-Tree
```

2) Navigate to the project directory:
```bash
cd Decision-Tree
```
3) Run the Python script:
```bash
python dectree.py
```

The script will load the dataset, train the decision tree model, and display the results including the decision tree plot, confusion matrix, and classification report.

## Visualizations
- Decision Tree Plot: Visualizes the trained decision tree model.
- Confusion Matrix: Shows the performance of the model by comparing the predicted and actual wine qualities.
- Classification Report: Provides metrics like precision, recall, F1-score, and support for each quality class.

## Results
The decision tree classifier provides insights into how the features affect the quality of red wine.
The model's performance is evaluated using a confusion matrix and a classification report to measure its accuracy and other related metrics.
