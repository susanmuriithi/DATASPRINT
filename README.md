# Financial Wellbeing in Kenya: A Multiclass ML Approach

## 📌Project Overview
This project investigates **who is struggling financially in Kenya and why**, using the **2024 FinAccess Household Survey** (20,871 adults across all 47 counties).  
We applied a **multiclass machine learning pipeline** to classify financial outcomes:  
- **Worsened**  
- **Stayed the Same**  
- **Improved**

The analysis highlights **income instability, literacy access, resilience, and regional disparities** as the strongest predictors of financial wellbeing.

---

## 📊 Dataset
- **Source**: 2024 FinAccess Household Survey  
- **Size**: 20,871 adults (cleaned to 20,863 after duplicate removal)  
- **Features**: 28 (demographics, income, savings, loans, mobile access, literacy, shock exposure)  
- **Target Distribution**:  
  - Worsened: 52.6%  
  - Stayed Same: 26.9%  
  - Improved: 20.5%  

**Data Cleaning**:
- Removed 5 duplicates  
- Filled nulls in `barriers_bank` (27.5%) with *Not Applicable*  
- Right‑skewed numerics transformed with Yeo‑Johnson  

---

## 🔎 Exploratory Data Analysis
- **Income drives outcomes**: Monthly income correlation `r = 0.72`  
- **Age vulnerability**: Adults 55+ show strongest negative correlation `r = -0.49`  
- **Noise in "Stayed Same"**: Overlaps both extremes, complicating classification  

---

## ⚙️ Methodology
1. **Feature Engineering**  
   - Derived resilience index, literacy scores, income per person, shock exposure  
   - Final feature set: 16  

2. **Baseline Models**  
   - Logistic Regression & Random Forest → Accuracy ~48–52%, F1 ~0.45  

3. **Noise Filtering**  
   - LDA separation identified stable vs. unstable zones  
   - Extracted 6,917 clean training samples  

4. **Final Model**  
   - Logistic Regression on clean dataset  
   - Stratified 70/30 split  
   - **Weighted F1 = 0.98, Accuracy = 98%**  

5. **Explainability**  
   - SHAP values showed **access_literacy, resilience_index, income** as dominant drivers  

---

## 📈 Model Performance
| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Worsened (0)   | 0.98      | 0.99   | 0.99     | 1,120   |
| Stayed Same (1)| 0.98      | 0.96   | 0.97     | 859     |
| Improved (2)   | 0.94      | 0.94   | 0.94     | 97      |
| **Weighted Avg** | **0.98** | **0.98** | **0.98** | 2,076   |

- **Confusion Matrix**: Only 31 misclassifications  
- **ROC-AUC**: ~0.71 on full dataset  

---

## 🔑 Key Findings
- **Income instability** → strongest predictor of financial deterioration  
- **Literacy access** → top SHAP driver across all classes  
- **Resilience index** → combines literacy, inclusion, shock readiness  
- **County disparities** → geographic inequity matters  
- **Age group (55+)** → most vulnerable  

---

## 🧭 Recommendations
### Policymakers
- Expand financial literacy programs (esp. adults 45+)  
- Invest in shock‑preparedness safety nets (insurance, emergency funds)  
- Reduce county‑level disparities via regional equity investment  

### Banks & Fintechs
- Design inclusive savings & lending products for low‑income households  
- Use resilience‑based scoring for credit decisions  
- Partner with mobile money providers to expand affordable access  

### NGOs & Development Partners
- Prioritise community‑level literacy & resilience training  
- Support income diversification programs  
- Target women & households with disabilities for tailored interventions  

---

## 🛠️ Tools & Environment
- **Languages**: Python 3.13  
- **ML Libraries**: scikit‑learn, LightGBM, CatBoost, XGBoost, SHAP  
- **Data Handling**: pandas, numpy, scipy  
- **Visualization**: matplotlib, seaborn, plotly  
- **Deployment/Workflow**: JupyterLab, Streamlit, FastAPI  

Environment is fully documented in [`environment.yml`](./name_base.txt).  
To recreate:
```bash
conda env create -f environment.yml
conda activate base
