# 🍷 Consumer Behavior & Marketing Intelligence Report

A specialized data engineering project using NumPy to analyze the relationship between household income and luxury product spending (Wine). This tool automates data cleaning and uses statistical thresholds to segment customer bases for targeted marketing campaigns.

## 📊 Key Analytical Insights (The Story)
* **Statistical Imputation:** Automatically handles missing data by calculating and injecting mean values into the Income dataset to maintain report accuracy.
* **Whale Segmentation:** Identifies high-value "Whales" by cross-referencing the 75th percentile of spenders with recent purchase activity (within 30 days).
* **Retention Alert System:** Detects "Lapsed VIPs"—high-spending customers who haven't purchased in over 60 days—providing actionable data for re-engagement campaigns.
* **Correlation Mapping:** Computes the Pearson correlation coefficient to determine if income is a primary driver of luxury sales.

## 🛠️ Tech Stack & Engineering
* **Core Engine:** Python 3.x
* **Primary Library:** NumPy (Used for high-speed array processing and statistical functions).
* **Data Processing:** Implements `np.genfromtxt` for tab-separated value (TSV) handling and `np.nanpercentile` for outlier detection.

## 🚀 Analytical Features
* **Income-to-Spend Ratios:** Calculates the percentage of consumer income directed toward specific product categories.
* **Percentile Thresholding:** Uses the 90th percentile to isolate top-tier spenders.
* **Automated Reporting:** Generates a formatted terminal report with real-time business insights.

---
## 👤 Developer
**Abhinav Adhikari**
*Aspiring full-Stack Developer & Aspiring Data Analyst*
* **Timeline:** Feb 2026
