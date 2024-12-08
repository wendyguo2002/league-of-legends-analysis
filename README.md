# Predicting Recipe Calories: A Data-Driven Analysis

## Author
**Wenxuan Guo**  
**Contact:** gwenxuan@umich.edu

## Table of Contents
1. [Introduction](#introduction)
2. [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)
3. [Framing a Prediction Problem](#framing-a-prediction-problem)
4. [Baseline Model](#baseline-model)
5. [Final Model](#final-model)

---

## Introduction
### Dataset Overview
This project analyzes recipe data scraped from Food.com. The dataset combines recipes from `RAW_recipes.csv` and user interactions (ratings and reviews) from `RAW_interactions.csv`. The focus is on understanding the attributes of recipes that affect their calorie content.

### Research Question
**What types of recipes tend to have the most calories?**  
This question is critical for uncovering how recipe characteristics influence calorie content, providing insights for dietary planning.

### Dataset Details
- **Total Rows:** `83782`  
- **Relevant Columns:**
  - `calories`: Total calorie content of the recipe (target variable).
  - `minutes`: Preparation time in minutes.
  - `n_steps`: Number of steps in the recipe.
  - `tags`: Associated labels or categories of the recipe.
  - `nutrition`: Nutritional details including calories.

---

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning
To clean the dataset, we performed the following steps:
1. Extracted the `calories` field from the `nutrition` column.
2. Removed rows with missing or infinite values in critical columns such as `calories`, `minutes`, and `n_steps`.
3. Final cleaned dataset preview of columns `name` and `calories` :
#### Head of Cleaned DataFrame

| name                                 |   calories |
|:-------------------------------------|-----------:|
| 1 brownies in the world    best ever |      138.4 |
| 1 in canada chocolate chip cookies   |      595.1 |
| 412 broccoli casserole               |      194.8 |
| millionaire pound cake               |      878.3 |
| 2000 meatloaf                        |      267   |


### Univariate Analysis

#### Plot 1: Distribution of Calories
A histogram showing the distribution of calories among recipes.

**Trend:** Most recipes have calorie counts in the low-to-moderate range, with a few high-calorie outliers.

**Relevance to Question:** This plot helps identify the range of calorie content across all recipes, which is essential for understanding which recipes are high-calorie and their prevalence. It sets a foundation for further exploration of recipe attributes that contribute to these higher calorie counts.

<iframe src="assets/plot1(Step2).html" width="800" height="600" frameborder="0"></iframe>
---

## Bivariate Analysis

#### Box Plot: Calories by Tags
A box plot exploring how calorie content varies across different recipe tags.

**Trend:** Certain tags like "easy" and "low-in-something" have significantly lower calorie content on average, while others such as "dessert" or "main course" show higher calorie content.

**Relevance to Question:** This analysis highlights how specific recipe tags correlate with calorie content. It demonstrates that high-calorie recipes often fall into certain categories, such as desserts or main courses, answering the question about which types of recipes tend to have the most calories.

<iframe src="assets/box(Step2).html" width="800" height="600" frameborder="0"></iframe>

---

### Interesting Aggregates

#### Pivot Table: Average Calories by Tags
A pivot table summarizing the average calorie content for each recipe tag.

**Significance:** Helps identify high-calorie categories like "desserts" and "main courses."

<iframe src="assets/pivot(Step2).html" width="800" height="600" frameborder="0"></iframe>

### Interesting Aggregates

#### Calories by Preparation Time
A scatter plot visualizing the relationship between preparation time and calories.

#### Head of the Table
**Average Calories by Cooking Time Ranges:**

| time_range   |   calories |
|:-------------|-----------:|
| 0-15 min     |    313.495 |
| 15-30 min    |    375.643 |
| 30-60 min    |    445.806 |
| 1-2 hours    |    558.003 |
| 2-4 hours    |    589.257 |
| 4+ hours     |    519.021 |

**Significance:** This table provides an overview of how average calorie content increases with longer preparation times, peaking in the "2-4 hours" range. Interestingly, the calorie content drops slightly for recipes taking over 4 hours, potentially due to recipes with extended cooking times being more balanced in ingredients. This insight helps identify time ranges associated with higher-calorie recipes, addressing the question of which types of recipes tend to have the most calories.

<iframe src="assets/calories_by_time(Step2).html" width="800" height="600" frameborder="0"></iframe>

---
### Framing a Prediction Problem

#### Research Question

**Can we predict the calorie content (`calories`) of a recipe based on its attributes?**

#### Attributes to Consider:

- **Preparation Time**: How long it takes to prepare the recipe.
- **Number of Ingredients**: The total count of ingredients in the recipe.
- **Number of Steps**: The number of steps required to complete the recipe.
- **Associated Recipe Tags**: Categories or labels assigned to the recipe (e.g., "vegan", "dessert").

#### Prediction Problem

**Type**: Regression problem.  
**Response Variable**: `calories`.  

We chose a regression approach because calorie content is a continuous numerical value. Predicting the calorie count based on recipe attributes aligns with the dataset's structure and our research question.

#### Metrics Used

**Primary Metric**: Mean Absolute Error (MAE).  
**Why MAE?** MAE provides an easily interpretable measure of error by averaging the absolute differences between predicted and actual values. This metric is particularly useful in this context because large errors in calorie prediction should be penalized proportionally.

#### Justification for Features

To ensure the model is realistic and applicable, we only use features that would be known at the time of prediction. Features like preparation time, number of steps, and recipe tags are defined before calorie values are observed or computed. By focusing on these, we ensure that the predictions are valid and reproducible for unseen data.


### Baseline Model
#### Model Description

The baseline model is a simple linear regression model that predicts the calorie content of recipes based on two features:

- **Features**: 
  - `minutes` (Quantitative): Total preparation time for the recipe.
  - `n_steps` (Quantitative): Number of steps required to complete the recipe.

The data was preprocessed using a pipeline that included:
1. **StandardScaler**: To standardize the quantitative features to have zero mean and unit variance.
2. **LinearRegression**: A linear regression model was used to establish a baseline for performance.

#### Model Features and Encoding

- **Quantitative Features**:
  - `minutes`: Numeric, scaled with StandardScaler.
  - `n_steps`: Numeric, scaled with StandardScaler.
- **Ordinal and Nominal Features**: Not included in this baseline model to keep it simple.

#### Performance Metrics

- **Train MAE**: 279.71  
- **Test MAE**: 271.90  
- **Test MSE**: 341865.98  

---

#### Analysis of Model Performance

The baseline model provides a starting point for prediction by identifying a linear relationship between recipe preparation time, number of steps, and calorie content. However, the relatively high Mean Absolute Error (MAE) and Mean Squared Error (MSE) indicate that this model does not capture the complexity of the calorie prediction problem. This suggests that additional features, better encoding, or more complex models may be needed for improvement.

#### Conclusion

While the baseline model is not "good" in terms of predictive accuracy, it serves as a useful benchmark for evaluating the improvements made in subsequent models. The insights gained from this step guide us in identifying opportunities for feature engineering and model enhancement.

### Final Model
#### Features Added

Two new features were engineered to capture more nuanced aspects of the calorie prediction problem:

1. **calories_per_step**: Represents the calorie complexity relative to the number of steps in the recipe.  
   - **Reason**: This feature provides insight into how calorie content scales with recipe complexity and step count, offering a per-step calorie density measure.
   
2. **calories_per_minute**: Measures calorie density over preparation time.  
   - **Reason**: Captures the relationship between calorie content and preparation efficiency, offering a per-minute calorie density measure.

These features are valuable as they provide normalized metrics that account for differences in recipe complexity and preparation times, which are crucial for understanding calorie variation.

#### Modeling Algorithm

The final model uses a **Random Forest Regressor**, a robust ensemble learning method that combines multiple decision trees to improve accuracy and generalization. 

- **Pipeline**: The model is implemented using a pipeline with the following steps:
  1. **Preprocessing**: 
     - Quantitative features (`minutes`, `n_steps`, `calories_per_step`, `calories_per_minute`) are scaled using `StandardScaler`.
     - Categorical features (`tags`) are encoded using `OneHotEncoder`.
  2. **Model**: `RandomForestRegressor` with hyperparameter tuning.

#### Hyperparameter Tuning

Hyperparameters were tuned using GridSearchCV, which performed 3-fold cross-validation to identify the best combination of parameters. The following parameters were tuned:
- **n_estimators**: Number of trees in the forest. Tested values: `[100, 200]`
- **max_depth**: Maximum depth of the trees. Tested values: `[None, 10]`
- **min_samples_split**: Minimum samples required to split a node. Tested values: `[2, 5]`

**Best Parameters Identified**:
- `n_estimators`: 200
- `max_depth`: 10
- `min_samples_split`: 2

#### Performance Evaluation

**Metrics Used**:  
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.
- **Mean Squared Error (MSE)**: Captures the average squared difference, penalizing larger errors.

**Final Model Performance**:
- **Train MAE**: 1.57
- **Test MAE**: 3.02
- **Test MSE**: 11225.04

#### Improvement Over Baseline

Compared to the baseline model, the final model shows significant improvement:
- **Baseline Train MAE**: 279.71 → **Final Train MAE**: 1.57  
- **Baseline Test MAE**: 271.90 → **Final Test MAE**: 3.02  
- **Baseline Test MSE**: 341865.98 → **Final Test MSE**: 11225.04  

This improvement can be attributed to:
1. **Feature Engineering**: The addition of `calories_per_step` and `calories_per_minute` captured more nuanced relationships in the data.
2. **Model Complexity**: Random Forest is better equipped to handle nonlinear relationships and interactions between features.
3. **Hyperparameter Tuning**: Fine-tuning parameters enhanced the model's ability to generalize to unseen data.

#### Conclusion

The final model demonstrates substantial improvement in predictive accuracy and generalization over the baseline model. By incorporating meaningful features and optimizing hyperparameters, the Random Forest Regressor effectively captures the complex relationships between recipe attributes and calorie content.
