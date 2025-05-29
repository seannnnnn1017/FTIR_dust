# FTIR-based Machine Learning Model for Predicting Soil TOC

## 🌱 Overview

This project presents a machine learning-based approach for predicting **Total Organic Carbon (TOC)** in soils and sediments using **Fourier-transform infrared spectroscopy (FTIR)**. The traditional method for TOC analysis is chemical testing, which is accurate but time-consuming and generates chemical waste. This project leverages FTIR as a rapid, non-destructive alternative, further enhanced by machine learning models.



## 📍 Motivation

- TOC is crucial for understanding **carbon storage**, which plays a key role in climate change mitigation.
- FTIR spectroscopy enables **non-invasive** TOC estimation, but raw spectral data are **complex** and subject to **sample variability**.
- Machine learning provides an **automated, scalable** solution to extract features, reduce noise, and improve predictive performance.

## 📊 Dataset

- **Location**: Tainan Science Park, Taiwan 🇹🇼
- **Samples**: 386 soil and sediment samples, depths up to 100 meters.
- **Data Composition**:
  - Raw FTIR spectra (4000–750 cm⁻¹)
  - TOC content from chemical analysis
- **Preprocessing**:
  - Outlier removal (via CDF) → **Fig. 6**
  - Dimensionality reduction (via PCA) → **Fig. 7**

> 📌 Reference Figures:
> - **Fig. 1**: Sampling Area Diagram  
>![Reference Figures Overview](assets\Sampling_Area_Diagram.png)
> - **Fig. 2**: Sample Preprocessing Flow  
>![Reference Figures Overview](assets\Preprocessing_Diagram.png)
> - **Fig. 3**: Raw FTIR Spectra Visualization  
>![Reference Figures Overview](assets\Raw_FTIR_Spectra_Visualization.png)
> - **Fig. 4**: Model Training Framework  
>![Reference Figures Overview](assets\Framework.png)
> - **Fig. 5**: Data Integration Framework  
>![Reference Figures Overview](assets\Data_Integration.png)
> - **Fig. 6**: TOC CDF & Outlier Detection  
>![Reference Figures Overview](assets\image.png)
> - **Fig. 7**: PCA Explained Variance  
>![Reference Figures Overview](assets\PCA.png)
> - **Fig. 8**: TOC Distribution Pre/Post Outlier Removal  
>![Reference Figures Overview](assets\Outlier.png)
> - **Fig. 9**: Model Prediction Performance Comparison  
>![Reference Figures Overview](assets\Prediction.png)
## 🧠 Machine Learning Models

Three regression models were evaluated:

- **SVR (Support Vector Regression)**  
- **XGBoost**  
- **AdaBoost**

### Processing Pipelines

1. **Raw Data**
2. **Outlier Removal**
3. **PCA-based Dimensionality Reduction**

> 🚀 Best performance achieved with SVR after PCA:
> - **MSE**: 0.0079  
> - **R²**: 0.693  
> - **Training Time**: 0.004 seconds

## 🧪 Experimental Results

| Model     | Preprocessing         | MSE     | R²     | Time     |
|-----------|------------------------|---------|--------|----------|
| SVR       | Raw                   | 0.0373  | 0.3378 | 0.049 s  |
| SVR       | Outlier Removed       | 0.0087  | 0.6618 | 0.033 s  |
| **SVR**   | **PCA (10 components)** | **0.0079** | **0.6930** | **0.004 s** |
| XGBoost   | PCA                   | 0.0092  | 0.6433 | 0.38 s   |
| AdaBoost  | PCA                   | 0.0086  | 0.6680 | 0.38 s   |

## 🧰 Tools & Environment

- **Python** 3.11  
- Libraries: `scikit-learn`, `xgboost`, `pandas`, `matplotlib`
- Hardware: Intel i5-12400, RTX 4080 SUPER, 48 GB RAM

## 🌍 SDGs Alignment

This work supports the following **UN Sustainable Development Goals**:

- 🥬 **SDG 2** – Zero Hunger  
- ♻️ **SDG 12** – Responsible Consumption and Production  
- 🌡️ **SDG 13** – Climate Action

## 🧭 Future Work

- Expand dataset across regions and soil types
- Explore **deep learning models** (e.g., CNNs, transformers)
- Integrate **real-time FTIR scanning** with on-site TOC prediction

## 📝 Citation

```bibtex
@article{chou2025ftir,
  title={A Machine Learning-Based Study on Predicting Soil and Sediment TOC Content Using FTIR Spectroscopy},
  author={Yung-Chen Chou, Xiang-Shun Yang, Yu-Siang Siang, Rou-Syuan Jiang, Kuo-Yen Wei and Li Lo},
  journal={Bachelor's Degree Program in AI, Feng Chia University},
  year={2025}
}
