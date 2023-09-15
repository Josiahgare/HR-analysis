## Predicting Employee Retention at salifort Motors
### Introduction
The HR at Salifort is concerned about the rate of employees leaving the company and wants to reduce it as much as possible. They have decided to use data-driven solution to help them get to the root of the cause and also take proactive measures towards retaining their employees.

The task is to build a model that predict employees who will leave the company. To do this, HR did a survey of their employees and provided the response to the data science team to run analysis on.  The deliverables include;
- One page executive summary[]
- A complete code notebook[]  

The `HR dataset` consist of 14,999 rows and 10 columns and analysis was carred out under a Plan,Analyze, Construct, and Execute (PACE) framework.

### Phase 1: Plan
In the plan phase, python was selected as the appropriate programming language for the task. Python libraries such as pandas, numpy, matpltlib, seaborn, sci-kit learn, xgboost, and pickle were imported into my jupyter notebook. The dataset was also loaded into the jupyter notebook for analysis to begin. Other aspect of the plan phase included data undertanding, exploration, and cleaning.

### Phase 2: Analyze
In the analyze plan, exploratory data analysis was carried out to find out the relationships between features and get insights through data visualization. The data was also prepared for modeling. *data visualization*

### Phase 3: Construct
In the construct phase, data preparation, model development, and model evaluation was carried out. For the purpose of this project regresson models and machine learning models were built to demonstrate proficiency.

For regression model, since we are dealing with a categorical and binary outcome feature "will leave vs will stay", the logistic model becomes the appropriate model to build. Data encoding and removal of outliers were done to aid the logistic model perform at its best. Lastly, seperating of predictor features and outcome feature was done as well as splitting the data into training and testing set for the model.

For machine learning model, decision tree model and random forest model were selected as appropriate model to build. The decision tree is a good enough model to build, however, it is prone to overfitting (getting used to the data used to train it and unable to predict accurately on new data). However, random forest model is not prone to overfitting becauses it uses an ensemble (aggregate of multiple trees) to train the model. In machine learning model, a technique known as `cross-validating` is used to ensure that the hyperparameters of the models are tuned to find the best hyprparameters for the model.

Also, feature engineering was done to tackle data leakage as well as improve model performance. Data leakage can result from a data that will not be used when the model is deployed.

Lastly, models were trained, tested, and evaluated using metrics such as classification report, accuracy, precision, recall, f1-score, and roc_auc. Visualization included `confusion matrix`, `decision tree split`, and `feature importance`.

### Phase 4: Execute
This include;
- defining evaluation metrics
- summary of model results
- conclusion, recommendation, and next steps.

NB: Due to the computational time of building a random forest model, they were run once and saved as pickle files and read back into the jupyter nootebook for analysis.
- Random forest model one[]
- Random forest model two[]
