# Social Media Addiction Level Classification (5 ML Models + Streamlit App)

## a. Problem Statement
Predict a user's **social-media addiction level** (Low, Moderate, High, Severe) from their usage habits, psychological
scores and demographics. This is a 4-class classification problem. Five classifiers are trained on the same data and compared
with Accuracy, AUC, Precision, Recall, F1 and MCC. A Streamlit app lets users upload a test CSV, choose a model and view the
metrics, a confusion matrix and a classification report.

## b. Dataset Description
- **Name:** Social Media Addiction & Mental Well-being (`social_media_addiction_mental_wellbeing.csv`).
- **Size:** 1,500 users, 30 columns (28 input features after dropping `User_ID`, plus the target).
- **Target:** `Addiction_Level` - Moderate 450, Low 420, High 360, Severe 270 (mildly imbalanced).
- **Numeric features (20):** Age, Daily_Usage_Hours, Platforms_Used_Count, Posts_Per_Week, Notifications_Per_Day, FOMO_Score,
  Social_Comparison_Score, Validation_Seeking_Score, Scroll_Without_Purpose, Anxiety_Score, Depression_Score, Loneliness_Score,
  Self_Esteem_Score, Sleep_Quality_Score, Sleep_Hours, Productivity_Loss_Score, Offline_Relationship_Quality,
  Physical_Activity_Hrs_Week, Screen_Free_Time_Hrs, Mental_Wellbeing_Score.
- **Categorical features (8):** Gender, Occupation, Relationship_Status, Primary_Platform, Late_Night_Usage, First_Check_Morning,
  Tried_To_Cut_Back, Failed_To_Cut_Back.
- **Missing values:** roughly 3-5% of most columns are blank.
- **Preprocessing (inside every saved model pipeline):** numeric -> median imputation + standard scaling; categorical ->
  most-frequent imputation + one-hot encoding. The app therefore accepts the raw columns directly.
- **Split:** 80/20 stratified (random_state=42): 1,200 train / 300 test. The test split is saved as `test_data.csv`.
- **Metric averaging:** Precision, Recall and F1 are macro-averaged over the 4 classes; AUC is one-vs-rest macro; MCC is the multiclass version.

## c. GitHub Repository Link
<FILL: https://github.com/<user>/<repo>>

Live app: <FILL: https://<app-name>.streamlit.app>

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8100 | 0.9527 | 0.8127 | 0.8106 | 0.8112 | 0.7439 |
| Decision Tree | 0.7400 | 0.9277 | 0.7548 | 0.7338 | 0.7418 | 0.6486 |
| KNN | 0.5300 | 0.7918 | 0.5667 | 0.5191 | 0.5332 | 0.3621 |
| Naive Bayes | 0.7400 | 0.9176 | 0.7536 | 0.7313 | 0.7394 | 0.6495 |
| Support Vector Machine | 0.7500 | 0.9364 | 0.7593 | 0.7405 | 0.7475 | 0.6629 |

Metrics are computed on the 300-row held-out test set (`test_data.csv`), printed by `train.py`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on every metric (Acc 0.81, AUC 0.95, MCC 0.74). Addiction level is driven mainly by a few roughly linear relationships, chiefly the mental-wellbeing score, so a linear model fits well and generalises cleanly. |
| Decision Tree | Good but behind logistic regression (Acc 0.74). It leans almost entirely on Mental_Wellbeing_Score (about 86% of its importance), so it is a coarse threshold model and loses information from the other scores. |
| KNN | Weakest (Acc 0.53, MCC 0.36). With 28 standardised and one-hot features, most of them noisy, distances become uninformative (curse of dimensionality). |
| Naive Bayes | Competitive for its simplicity (Acc 0.74, AUC 0.92). Many scores are roughly Gaussian and only weakly correlated, so the independence assumption is not badly violated; it matches the tree on accuracy. |
| Support Vector Machine | Second best by AUC (0.94) and MCC (0.66). The RBF kernel handles the many features well but gains nothing over the linear model because the class boundaries are close to linear. |

**Overall winner:** **Logistic Regression** - highest Accuracy, AUC, Precision, Recall, F1 and MCC on the test set.

**Caveat:** `Mental_Wellbeing_Score` is an outcome of heavy use rather than a cause, and it carries most of the predictive signal. It is kept
because it is part of the dataset, but a model restricted to behavioural inputs would score lower.

## Project Structure
```
app.py            Streamlit app (loads saved models, never retrains)
train.py          preprocessing + trains the 5 models, writes model/*.joblib and test_data.csv
requirements.txt
test_data.csv     held-out test split (features + Addiction_Level)
model/            logistic_regression / decision_tree / knn / naive_bayes / support_vector_machine .joblib
social_media_addiction_mental_wellbeing.csv   raw dataset
```

## Run Locally
```
pip install -r requirements.txt
python train.py social_media_addiction_mental_wellbeing.csv
streamlit run app.py
```
