# CSE3141 – Predictive Analytics: Course Project
**Manipal University Jaipur | Department of Computer Science & Engineering**

[![Live Demo on Vercel](https://img.shields.io/badge/Live_Dashboard-Vercel-black?style=for-the-badge&logo=vercel)](https://predictive-analytics-bike-sharing.vercel.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/A-k-s-h-a-t-2-0/predictive-analytics-bike-sharing)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FA-k-s-h-a-t-2-0%2Fpredictive-analytics-bike-sharing)

---

## 🌟 Project Title
**Predictive Demand Modeling for Urban Bike Sharing using Generalized Linear Models (Poisson & Negative Binomial Regression)**

### 🔗 Live Interactive Dashboard
Try the live model demo with real-time demand calculation and visual analytics:
👉 **[https://predictive-analytics-bike-sharing.vercel.app](https://predictive-analytics-bike-sharing.vercel.app)**

---

## 📁 Repository Structure
```
predictive analysis project/
├── index.html              # Interactive Web Dashboard (Hosted on Vercel)
├── vercel.json             # Vercel deployment configuration
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

---

## 🏆 Model Benchmark Results

| Model | Test RMSE (Lec 31) | Test MAE (Lec 32) | $R^2$ Score | Negative Predictions ($< 0$) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Linear Regression (OLS)** | 92.767 | 69.937 | 0.728 | **324 (Invalid)** | Violates Non-negativity |
| **2. Poisson Regression (Lec 20)** | 88.698 | 63.065 | 0.752 | **0 (Valid)** | Strictly Non-negative |
| **3. Negative Binomial (Lec 21)** | 152.036 | 88.076 | 0.270 | **0 (Valid)** | Models Overdispersion |
| **4. Tuned Poisson (GridSearchCV)** | **87.836** | **62.302** | **0.758** | **0 (Valid)** | **Top Performer** |

---

## 🚀 How to Run Locally

### Option A: Using Jupyter Notebook (Submission Ready)
1. Open `predictive_analytics_bike_sharing.ipynb` in VS Code or JupyterLab.
2. All cells are already pre-run with outputs, graphs, and summary tables intact.

### Option B: Using the Command Line
Run the standalone analysis script:
```bash
python run_analysis.py
```
This generates all figures and prints the complete statistical summary to your terminal.
