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
