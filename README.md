# Predicting Gender Pay Gap Trends Across UK Industries

## Project Overview

This project investigates whether changes in the UK gender pay gap can be predicted using historical employer-level gender pay gap data, with particular attention to the fashion and retail sectors.

Using annual UK Gender Pay Gap reporting data, the project develops and evaluates machine learning models to predict changes in employer-reported gender pay gaps between consecutive reporting years.

The analysis focuses on identifying patterns, evaluating predictive performance, and examining differences across industry sectors.

## Research Objectives

- Analyse historical trends in the UK gender pay gap.
- Prepare an employer-level dataset covering consecutive reporting years.
- Investigate sector-level differences, particularly in fashion and retail.
- Develop predictive models to estimate annual changes in the gender pay gap.
- Compare machine learning models against baseline approaches.
- Evaluate model performance using regression and classification metrics.
- Examine model interpretability and sector-level variation.

## Dataset

The project uses UK Gender Pay Gap employer reporting data covering reporting years 2017–18 to 2025–26.

The data is processed to construct an employer-year panel and identify consecutive-year employer observations.

The prediction target is the annual change in the reported median gender pay gap:

`target_change = next_year_gap - current_year_gap`

A positive value represents an increase in the reported gap, while a negative value represents a decrease.

## Methodology

The project follows these main stages:

1. Data collection and integration
2. Data cleaning and preprocessing
3. Employer-level panel construction
4. Exploratory data analysis
5. Feature engineering
6. Model development and tuning
7. Temporal validation and final holdout evaluation
8. Model interpretation using SHAP and permutation-based methods
9. Sector-level performance analysis

## Models Evaluated

The following models were implemented:

- NoChange baseline
- MeanReversion baseline
- Ridge Regression
- Random Forest
- HistGradientBoosting

Models were evaluated using a temporal validation strategy to reduce the risk of information leakage between reporting years.

## Evaluation Metrics

Regression performance was assessed using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Classification analysis additionally examined the ability to identify widening, stable and narrowing gender pay gap trends.

## Key Results

The final holdout evaluation produced the following results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| NoChange | 4.131 | 7.942 | -0.003 |
| MeanReversion | 4.711 | 7.821 | 0.027 |
| Ridge | 4.447 | 7.903 | 0.007 |
| Random Forest | 4.031 | 7.432 | 0.122 |
| HistGradientBoosting | 4.011 | 7.671 | 0.064 |

Random Forest achieved the lowest RMSE and highest R², while HistGradientBoosting achieved the lowest MAE.

The results indicate some predictive signal, although overall predictive performance remains limited.

## Project Structure

```text
gpg_project/
│
├── data/
│   └── raw/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── EDA_1.ipynb
│   ├── EDA_2.ipynb
│   └── EDA_3.ipynb
│
├── outputs/
│   └── main/
│       ├── figures/
│       └── tables/
│
├── analysis.py
├── config.py
├── data_prep.py
├── download_data.py
├── make_synthetic_data.py
├── modelling.py
├── run_pipeline.py
├── run_all.sh
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SHAP
- Jupyter Notebook

## Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the main pipeline:

```bash
python run_pipeline.py
```

The pipeline generates analysis outputs, model evaluation results, visualisations and supporting tables.

## Limitations

- Employer-reported gender pay gaps do not measure equal pay for identical work.
- Predictive performance is limited and should be interpreted cautiously.
- Sector-level comparisons may be affected by differences in employer representation.
- Predictive associations do not establish causal relationships.
- Results depend on the quality, coverage and consistency of the underlying reporting data.

## Conclusion

This project applies predictive analytics to historical UK gender pay gap data to investigate whether employer-level changes can be anticipated using machine learning.

The findings demonstrate the potential and limitations of predictive modelling in this area, while highlighting the importance of temporal validation, model interpretation and sector-level analysis.