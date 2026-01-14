**#Suitable Crop Prediction via Multi-Class Classification (Machine Learning)**

**## Problem Statement**

Predict the optimal crop for a farm using soil macronutrients and environmental parameters through a multi-class classification approach.

**## Problem Context**

Agriculture is highly sensitive to soil and environmental variability, making crop selection a non-trivial classification problem. Differences in soil macronutrients, climatic conditions, and environmental factors directly impact crop suitability and yield. This project formulates crop recommendation as a **multi-class classification task**, using soil health indicators and environmental features to learn patterns associated with optimal crop growth. By applying supervised machine learning techniques to predict the most suitable crop for given farm conditions, the model enables data-driven agricultural decision-making, improved yield potential, and efficient resource utilization.

**## Dataset**

The dataset contains soil nutrient and environmental parameters used to predict the most suitable crop for cultivation. It is sourced from a publicly available Kaggle dataset.

Source: Kaggle – **Crop Recommendation Dataset**

**Input Features:**
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall
- 
**Target Variable:**
  
- Label = variety of crops

[Dataset Features] (images./Crop_Recommendation_Dataset_Features.png)

**## Notebook (Google Colab)**

This project notebook can be executed directly in Google Colab without any local setup.

[Open in Google Colab]

Copy of Suitable Crop Prediction via Multi-Class Classification.ipynb - Colab

**## Technical Approach**
- Data cleaning and visualization
- Feature engineering
- Model training and comparison  
- Model evaluation using **weighted-average** multi-class classification metrics
- Applied scaling and preprocessing exclusively on the training set to prevent data leakage and preserve test set integrity.
- Focused on class-balanced evaluation metrics and confusion matrix analysis rather than binary-centric ROC curves to ensure meaningful performance assessment.

**## Tech Stack**

Python, Pandas, Scikit-learn, Seaborn, Matplotlib

**## Machine Learning Models**
- Logistic Regression  
- Decision Tree
- Random Forest Classifier  
- K-Nearest Neighbors (KNN) 
- Naive Bayes
- Ensemble Methods (Gradient Boosting and Ada Boost Model)

**## Model Evaluation**

Evaluation was performed using the following metrics:
- Precision (Weighted)
- Accuracy 
- Recall (Weighted) 
- F1-Score (Weighted) 

**## Results**
- Achieved a maximum **accuracy of 99.55%** on the **test dataset** using both **Random Forest Classifier** and **Naive Bayes**, indicating strong predictive capability.
- **Random Forest Classifier** demonstrated consistently high performance across **Precision (99.55%)**, **Recall (99.55%)**, and **F1-score (99.55%)**, making it the most reliable model among those evaluated.
- 
[Performance Scores]:

{With Preprocessing} (images./Crop_Prediction_Performance_of_Models.png)
- Random Forest was **selected** due to its superior accuracy, robustness to overfitting, and consistent performance across all classes compared to other evaluated models. It enabled interpretable predictions through feature importance analysis, a capability not available in probabilistic baseline models.

**##Model Evaluation Rationale**

This project addresses a multi-class classification problem, where traditional binary evaluation metrics such as a single ROC curve are not directly applicable. ROC curves require decomposition into multiple one-vs-rest comparisons in multi-class settings, which can reduce interpretability and obscure class-level performance.

To ensure meaningful and transparent evaluation, the model was assessed using:

•	**Accuracy** (for overall performance comparison)

•	**Precision, Recall, and F1-score (weighted)** to account for class distribution

•	**Confusion Matrix** for per-class error analysis

- **Weighted averaging** was chosen to account for **class imbalance** and to ensure that performance across all classes was fairly represented.

**## Project Structure**

crop-recommendation/
├── data/
│   └── crop_data.csv                          # Soil & environmental dataset
├── notebooks/
│   └── crop_recommendation.ipynb  # EDA, feature analysis & modeling
├── src/
│   ├── preprocessing.py                     # Data cleaning, scaling, encoding
│   ├── train.py                                    # Model training logic
│   └── evaluate.py                              # Model evaluation (accuracy, F1, recall)
├── requirements.txt                            # Project dependencies
└── README.md                               # Project documentation

**## Setup & Run**
1. Clone the repository:
  ```bash
  git clone https://github.com/your-username/<project-repo-name>.git
  cd <project-repo-name>

2. Create and activate a virtual environment to isolate project dependencies.
    python -m venv venv
    source venv/bin/activate    #macOS/Linux
    venv\Scripts\activate         # Windows

3. Install dependencies:
    pip install -r requirements.txt

4. Run the notebook:
    jupyter notebook

5. (Recommended) Run on Google Colab by uploading the notebook/clicking on ‘Open on Colab’ button.


**## Key Learnings**
- Applied correlation analysis exclusively on numerical features, excluding nominal categorical data, as correlation is meaningful only for continuous data. This step was used purely for **exploratory visualization** to identify linear associations and multicollinearity prior to model training.

[Correlation Matrix Applicability] (images./Crop_Prediction_Key_Learning_Correlation.png)

- All preprocessing steps (such as feature scaling) were applied **only on the training dataset** to preserve test set integrity and prevent data leakage.

**## Disclaimer**
Model outputs are dependent on the underlying dataset and assumptions and may not generalize across regions, seasons, or real-world conditions due to environmental variability, data coverage limitations, and potential sampling bias.


