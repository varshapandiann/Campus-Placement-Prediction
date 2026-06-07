# Project Report - AD Lab - Predicting Campus Placement Outcomes

## 1. Introduction

### 1.1 Overview

Campus placement plays an important role in shaping students' career trajectories, serving as a crucial transition from academia to the professional world. A student’s employability is often determined by a combination of academic performance, technical expertise, soft skills, and practical experience. As competition for jobs intensifies, institutions and students alike seek data-driven approaches to enhance placement success. Machine learning has emerged as a powerful tool for predictive analysis in various domains, including education and employment forecasting.

This study employs ML techniques to predict students' placement outcomes based on a dataset containing academic achievements, internship experiences, certifications, and extracurricular activities. Initially, a Support Vector Machine (SVM) model was implemented due to its strong classification capabilities, particularly in high-dimensional data. However, due to SVM’s susceptibility to overfitting and computational inefficiencies, a Random Forest Classifier was introduced as an alternative to improve model generalization and interpretability.

This report presents a comparative study of the two models, examining their predictive performance, overfitting tendencies, and practical usability. The research encompasses data collection, preprocessing, feature selection, model training, hyperparameter tuning, performance evaluation, and visualization to derive meaningful insights. By leveraging machine learning, this study aims to provide institutions with actionable intelligence to refine placement strategies and enhance student preparedness.

### 1.2 Problem Statement

Despite structured placement training programs and career development initiatives, many students face challenges in securing employment. Educational institutions often struggle to pinpoint the key determinants of placement success, making it difficult to tailor interventions effectively. The primary challenges in placement prediction include: 

* **Identifying Key Influencing Factors:** Determining which attributes—academic performance, internships, certifications, extracurricular involvement, or soft skills—play the most significant role in job placement outcomes.
* **Handling Imbalanced Data:** The placement dataset is often imbalanced, with a smaller proportion of students securing jobs compared to the total number of applicants. This imbalance can impact model performance and prediction reliability.
* **Selecting an Optimal Machine Learning Model:** Different ML models vary in their ability to handle classification tasks, generalize well, and provide interpretability. Finding the most suitable model requires comparative evaluation.
* **Ensuring Interpretability of Predictions:** Placement prediction models should not only be accurate but also interpretable, enabling institutions to derive meaningful insights and guide students in improving their employability.

### 1.3 Objectives

The primary objectives of this research are:
* **Analyze Placement Data:** Investigate patterns and trends in student placement records to identify factors contributing to successful job acquisition.
* **Preprocess and Clean Data:** Implement data cleaning, normalization, and feature selection techniques to ensure high-quality input for ML models.
* **Train and Compare Machine Learning Models:** Develop multiple ML models, beginning with SVM and subsequently implementing Random Forest, to determine their predictive capabilities and performance differences.
* **Interpret Feature Importance:** Utilize model-based feature importance analysis to highlight the most influential factors affecting placement outcomes.
* **Visualize Results for Decision-Making:** Employ graphical representations, including decision tree diagrams and feature importance plots, to enhance interpretability and facilitate informed decision-making.
* **Propose Placement Improvement Strategies:** Based on the findings, recommend data-driven strategies for students and educational institutions to optimize career preparedness and placement success rates.

By addressing these objectives, this study aims to bridge the gap between academic training and industry requirements, ensuring that students are better equipped to secure employment opportunities.

## 2. Literature Review

### 2.1 Existing Research on Placement Prediction

Several studies have explored the application of machine learning techniques to predict employability based on academic and non-academic factors.  Researchers have leveraged various classification algorithms to improve placement prediction accuracy. Some common approaches include:
* **Support Vector Machines (SVM):** Used for high-dimensional classification problems where clear decision boundaries need to be established. Studies have demonstrated its effectiveness in classifying students based on placement status.
* **Random Forest and Gradient Boosting:** Ensemble methods like Random Forest and XGBoost have been widely adopted due to their ability to handle complex, nonlinear relationships and provide better interpretability.
* **Deep Learning Models (LSTMs and CNNs):** Applied in scenarios requiring pattern recognition from sequential or unstructured data, such as analyzing student academic progress over time.
* **Natural Language Processing (NLP)-Based Resume Analysis:** Some research has explored NLP-based techniques for parsing resumes and predicting candidate suitability for specific job roles based on textual features.

### 2.2 Review of Existing Research Papers

Several research papers have contributed to the understanding of placement prediction using machine learning models:
* **Sharma & Agrawal (2021):** Implemented an SVM-based model for student placement prediction and found that the model achieved 80% accuracy but struggled with overfitting in small datasets.
* **Gupta et al. (2020):** Compared Decision Trees, Random Forest, and Neural Networks for employability prediction and concluded that ensemble models, particularly Random Forest, provided better generalization and feature interpretability.
* **Patel & Kumar (2019):** Analyzed academic performance, certifications, and extracurricular activities, emphasizing that non-academic factors significantly influence placement success.
* **Singh et al. (2022):** Used a hybrid machine learning approach combining SVM and Gradient Boosting to achieve higher accuracy while addressing class imbalance issues.

These studies highlight the strengths and limitations of different ML approaches while reinforcing the need for a model that balances accuracy, interpretability, and generalizability.

### 2.3 Limitations of Existing Approaches

While these studies provide valuable insights, certain gaps remain unaddressed:
* **Overlooking Non-Academic Factors:** Many studies focus solely on academic performance, ignoring key elements like technical certifications, extracurricular involvement, and soft skills.
* **Lack of Real-World Hiring Trends:** Datasets in research papers often do not capture dynamic industry hiring patterns, limiting the model's adaptability.
* **Data Imbalance Issues:** Many studies use datasets where the number of placed students significantly outweighs unplaced students, or vice versa, leading to biased predictions.

Our study aims to address these limitations by incorporating a more holistic dataset that includes academic records, internships, certifications, and extracurricular involvement. By implementing both SVM and Random Forest classifiers, we analyze their strengths and weaknesses, ultimately recommending the most effective approach for placement prediction.

### 2.4 Description of the Dataset Used

The dataset for this study was collected from student placement records and contained multiple features that influence employability. The key attributes included:
* **Academic Performance:** Cumulative GPA, grades in specific subjects, and academic trends.
* **Internships & Job Experience:** Number and quality of internships completed, industry-recognized certifications obtained.
* **Demographic Information:** Basic details such as age, gender, and location.
* **Placement Outcome:** A binary label (Placed/Not Placed) indicating employment status.

The dataset was preprocessed to handle missing values, normalize numerical attributes, and apply feature selection techniques to improve model performance. By utilizing a diverse dataset and comparing different machine learning techniques, our study provides a comprehensive analysis of placement prediction, bridging the gaps identified in existing research.

## 3. Methodology
This section outlines the systematic approach followed in data collection, preprocessing, model development, and performance evaluation for student placement prediction. The methodology follows a structured pipeline, starting from raw data acquisition to final model comparison.

### 3.1 Data Collection and Preprocessing
To develop an accurate placement prediction model, data was collected from student placement records over multiple academic years. The dataset comprised a diverse range of attributes that influence employability, including academic achievements, technical skills, and extracurricular participation.

#### Dataset Description
The dataset contained **X** records of students with the following key attributes:

- **Academic Performance:**
  - Cumulative GPA
  - Grades in core subjects (Mathematics, Programming, Communication Skills, etc.)
  - Trend of academic performance over semesters

- **Internships and Job Experience:**
  - Number and type of internships completed
  - Certifications in relevant technical and soft skills (e.g., AWS, Google Cloud, Coursera, Udemy courses)
    
- **Demographic Information:**
  - Age, gender, location
    
- **Placement Outcome:**
  - A binary classification label (1 = Placed, 0 = Not Placed)
 
####  Data Preprocessing Steps
To ensure model accuracy and reliability, the following preprocessing techniques were applied:

- **Handling Missing Values**
  - **Numerical Attributes:** Missing values in GPA and grades were replaced using mean imputation to preserve overall data distribution.
  - **Categorical Variables:** Department, certification status, and internship details were filled using mode imputation (most frequent category).

- **Data Normalization**
  - To standardize numerical features and improve model efficiency, Min-Max Scaling was applied, particularly for models sensitive to feature magnitude, like SVM.

- **Feature Engineering & Selection**
  - Highly correlated variables were removed to reduce multicollinearity and avoid redundant features.
  - Recursive Feature Elimination (RFE) was used to identify the most relevant predictors for placement.

- **Data Splitting**
  - The dataset was split into 80% training and 20% testing subsets to evaluate model performance on unseen data.

### 3.2 Support Vector Machine (SVM) Implementation

#### Initial Model Development
SVM was chosen as the first model due to its effectiveness in high-dimensional classification problems.

- **Kernel Selection**
  - A **linear kernel** was initially applied but failed to capture complex decision boundaries.
  - The **Radial Basis Function (RBF)** kernel was later selected to introduce non-linearity and improve performance.

- **Hyperparameter Tuning**

  - `GridSearchCV` was used to optimize:

    -   **C (Regularization parameter):** Controls the trade-off between low error and model complexity.
    -   **Gamma:** Defines the influence of individual training samples on decision boundaries.

### 3.3 Observations from SVM Model

Despite achieving a high accuracy of **81.39%**, the SVM model exhibited:

-   **Overfitting:** Performed exceptionally well on training data but struggled to generalize.
-   **Sensitivity to Outliers:** Noisy data caused misclassifications, affecting recall.
-   **Computational Complexity:** Training and tuning became expensive as dataset size increased.

### 3.4 Switching to Random Forest Classifier

Random Forest was explored due to its robustness and ability to reduce overfitting.

#### Why Random Forest?

-   Reduces overfitting through bagging.
-   Provides feature importance analysis.
-   Handles both numerical and categorical variables.

#### Implementation Steps

- Hyperparameter Tuning
  - Optimized using `GridSearchCV`:

    -   `n_estimators = 100`
    -   `max_depth`
    -   `criterion = "gini"`

- Feature Importance Analysis

  - Random Forest identified the most influential predictors affecting placement success.

### 3.5 Comparative Analysis: SVM vs Random Forest

| Metric | SVM | Random Forest |
| :--- | :---: | :---: |
| Accuracy | 81.39% | 79.07% |
| Precision | 81% | 78% |
| Recall | 81% | 79% |
| F1-Score | 80% | 78% |
| Overfitting | Yes | No |
| Training Time | High | Moderate |

#### Key Findings

- SVM achieved higher accuracy but suffered from overfitting.
- Random Forest generalized better and was more robust.
- Random Forest produced useful feature importance scores.
- SVM had slightly higher precision, while Random Forest had better recall.
- Random Forest required less computation.

#### SVM Results

-   **Accuracy:** `0.813953488372093`

``` text
              precision recall f1-score support

0                 0.75   0.50    0.60      12
1                 0.83   0.94    0.88      31

accuracy                           0.81    43
macro avg         0.79   0.72    0.74      43
weighted avg      0.81   0.81    0.80      43
```

#### Random Forest Results

-   **Accuracy:** `0.7906976744186046`

``` text
              precision recall f1-score support

0                 0.67   0.50    0.57      12
1                 0.82   0.90    0.86      31

accuracy                           0.79    43
macro avg         0.75   0.70    0.72      43
weighted avg      0.78   0.79    0.78      43
```

### 3.6 Diagram Representations

-   Random Forest Decision Tree Representation
  <img width="1005" height="516" alt="image" src="https://github.com/user-attachments/assets/74e064bb-1102-4976-a9f3-9db6664cfd89" />

-   Feature Importance
  <img width="927" height="513" alt="image" src="https://github.com/user-attachments/assets/676629e4-c2ee-4c63-aa26-2a0314a998e7" />


## 4. Conclusion and Future Work

### 4.1 Summary of Findings

- SVM achieved strong accuracy but was prone to overfitting.
- Random Forest generalized better and was easier to interpret.
- Internships, certifications, and non-academic achievements significantly influenced placement success.
- Feature importance analysis helped identify impactful factors.

### 4.2 Future Enhancements

- Implement XGBoost.
- Explore deep learning models (MLPs/CNNs).
- Integrate NLP-based resume parsing.
- Handle class imbalance using SMOTE or cost-sensitive learning.
- Incorporate real-world industry trends and recruiter preferences.
