

The  `/scripts_analysis` folder contains scripts for the data analysis involved in the case study, model comparisons, and other supplementary analyses presented in the paper.

---

#### **Case Study Workflow**  
For the case study analysis, execute scripts in the following order:  
- `/case_study_data` for data preparation and loading of case study ; 
- `/case_study_run` for running tests of case study;
- `/case_study_evaluation` for evaluating model performance metrics.

---
#### **Model Comparisons**  
- `/model_comparison` compares ExpressRM against competing models.  
---

#### **Performance Visualization**  
- `plot_model_performance.R` visualizes model performance metrics.
- `plot_model_performance_1000_epochs.R` visualizes results after 1000 epochs of training.
---

#### **Feature Importance Analysis**  
- `/shap` calculates SHAP values for model interpretation and feature contribution.
