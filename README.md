# Predictive Analytics: Course Project
**Department of Computer Science & Engineering | Manipal University Jaipur**

[![Live Demo on Web](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-2ea44f?style=for-the-badge&logo=github)](https://a-k-s-h-a-t-2-0.github.io/predictive-analytics-bike-sharing/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/A-k-s-h-a-t-2-0/predictive-analytics-bike-sharing)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FA-k-s-h-a-t-2-0%2Fpredictive-analytics-bike-sharing)

---

## 🌟 Project Title
**Predictive Demand Modeling for Urban Bike Sharing using Generalized Linear Models (Poisson & Negative Binomial Regression)**

### 🔗 Live Interactive Dashboard
Try the live model demo with real-time demand calculation and visual analytics:
👉 **[https://predictive-analytics-bike-sharing.vercel.app](https://predictive-analytics-bike-sharing.vercel.app)**  
*(Backup mirror: [GitHub Pages](https://a-k-s-h-a-t-2-0.github.io/predictive-analytics-bike-sharing/))*

---

## 📁 Repository Structure
```
predictive analysis project/
├── index.html              # Interactive Web Dashboard (Hosted live on Vercel)
├── vercel.json             # Vercel deployment configuration
├── data/
│   ├── hour.csv            # Hourly Capital Bikeshare records (17,379 rows, 17 columns)
│   ├── day.csv             # Daily aggregated records
│   └── Readme.txt          # UCI Dataset documentation
├── figures/                # Exported high-resolution visualization charts
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

## 📊 Analytical Pipeline & Methodology

| Pipeline Stage | Technique / Focus | Implementation in this Project |
| :--- | :--- | :--- |
| **1. Project Scope & Formulation** | Problem Formulation & Ethics | Formulated bike fleet rebalancing and equitable urban accessibility. |
| **2. Data Ingestion & Quality Audit** | Data Ingestion & Missing Value Check | Real-world Capital Bikeshare / UCI sensor logs; verified 0 missing records. |
| **3. Preprocessing & Outliers** | IQR Outlier Detection & Boxplots | Computed IQR ($Q_3 - Q_1 = 241$); identified 505 peak commuter hours ($>642.5$). |
| **4. Statistical Profiling** | Descriptive Stats & Overdispersion Test | Calculated Mean, Median, Mode, Variance, Z-Scores; proved $\text{Var}/\text{Mean} = 173.66$. |
| **5. Exploratory Visualizations** | 5 Core Analytical Charts | Histogram, Scatter Plot, Seasonal Bar Chart, Weather Pie Chart, and Heatmap. |
| **6. Feature Engineering** | Domain Flags & Cyclical Encodings | Created `rush_hour` commuter flag, cyclical $\sin/\cos$ hour waves, and one-hot encodings. |
| **7. Predictive Modeling (GLM)** | Generalized Linear Models | Compared OLS Linear Baseline vs. Poisson Regression ($e^{X\beta}$) vs. Negative Binomial. |
| **8. Validation & Optimization** | Holdout Evaluation & Hyperparameter Tuning | 80/20 train/test split, RMSE, MAE, and 5-Fold `GridSearchCV` parameter optimization. |

---

## 🏆 Model Benchmark Results

| Model | Test RMSE | Test MAE | $R^2$ Score | Negative Predictions ($< 0$) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Linear Regression (OLS)** | 92.767 | 69.937 | 0.728 | **324 (Invalid)** | Violates Non-negativity |
| **2. Poisson Regression** | 88.698 | 63.065 | 0.752 | **0 (Valid)** | Strictly Non-negative |
| **3. Negative Binomial** | 152.036 | 88.076 | 0.270 | **0 (Valid)** | Models Overdispersion |
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
