# CSE3141 – Predictive Analytics: Course Project
**Manipal University Jaipur | Department of Computer Science & Engineering**

## Project Title
**Predictive Demand Modeling for Urban Bike Sharing using Generalized Linear Models (Poisson & Negative Binomial Regression)**

---

## 📁 Repository Structure
```
predictive analysis project/
├── data/
│   ├── hour.csv            # Hourly Capital Bikeshare records (17,379 rows, 17 columns)
│   ├── day.csv             # Daily aggregated records
│   └── Readme.txt          # UCI Dataset documentation
├── figures/                # Exported high-resolution visualization charts (Lectures 6, 9, 36)
│   ├── 01_boxplot_outliers_lec6.png
│   ├── 02_histogram_lec9.png
│   ├── 03_scatter_plot_lec9.png
│   ├── 04_bar_chart_lec9.png
│   ├── 05_pie_chart_lec9.png
│   ├── 06_heatmap_lec9.png
│   └── 07_actual_vs_predicted.png
├── make_notebook.py        # Automated notebook builder script
├── run_analysis.py         # Standalone end-to-end Python execution script
├── predictive_analytics_bike_sharing.ipynb  # Fully executed, interactive Jupyter Notebook
└── README.md               # Project documentation and execution instructions
```

---

## 🚀 How to Run the Project

### Option A: Using Jupyter Notebook (Recommended for submission)
1. Open the workspace in VS Code or launch Jupyter:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
2. Open `predictive_analytics_bike_sharing.ipynb`.
3. All cells are already pre-executed with plots and tables rendered. You can also re-run the notebook anytime with `Kernel -> Restart & Run All`.

### Option B: Using the Command Line
Run the standalone analysis script:
```bash
python run_analysis.py
```
This will print the descriptive statistics, overdispersion test, model comparisons, and save all 7 publication-ready charts to the `figures/` directory.

---

## 📊 Summary of Lecture Topics Covered

| Lecture Number | Syllabus Requirement | Implementation in this Project |
| :--- | :--- | :--- |
| **Lec 1–4** | Predictive Analytics Scope & Ethics | Formulated bike rebalancing problem and ethical access across urban areas. |
| **Lec 5** | Data Collection Methods | Real-world Capital Bikeshare / UCI hourly sensor logs. |
| **Lec 6** | Preprocessing, Missing Values & Outliers | Missing value audit; IQR outlier detection & Boxplots across seasons. |
| **Lec 7–8** | Mean, Median, Mode, Variance, Z-Scores | Computed summary table; detected $|Z|>3$; proved overdispersion ($\text{Var}/\text{Mean} = 173.66$). |
| **Lec 9** | Visualizations | Histogram, Scatter Plot, Bar Chart, Pie Chart, and Heatmap. |
| **Lec 10–11** | Feature Engineering & Selection | Rush hour flag, cyclical hour encoding ($\sin, \cos$), one-hot dummy variables. |
| **Lec 19–21** | Generalized Linear Regression | OLS Baseline vs. Poisson Regression ($e^{X\beta}$) vs. Negative Binomial. |
| **Lec 30–36** | Validation, Metrics & Tuning | 80/20 train/test split, RMSE, MAE, and 5-Fold `GridSearchCV` parameter tuning. |
