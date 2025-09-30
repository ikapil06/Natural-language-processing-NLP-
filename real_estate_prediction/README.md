# 🏠 Egyptian Real Estate Price Prediction Analysis

A comprehensive machine learning project for predicting real estate prices in Egypt using multiple algorithms and advanced feature engineering techniques.

## 📋 Project Overview

This project analyzes Egyptian real estate listings data to build predictive models for property prices. The analysis includes data cleaning, exploratory data analysis, feature engineering, and comparison of multiple machine learning algorithms.

## 🎯 Key Results

- **Best Model**: Gradient Boosting Regressor with hyperparameter tuning
- **Performance**: **R² = 0.3040** (30.4% variance explained)
- **Improvement**: **+20.1%** over baseline linear regression
- **Accuracy**: Mean Absolute Error of ±4.7 million EGP

## 📊 Dataset

- **Source**: Egyptian real estate listings
- **Original Size**: 19,924 properties
- **Final Clean Dataset**: 16,616 properties (85.1% retention)
- **Features**: 11 columns including price, size, bedrooms, bathrooms, location, etc.

## 🔍 Analysis Pipeline

1. **Data Loading & Cleaning**
   - Fixed file path issues
   - Cleaned numeric columns
   - Handled missing values
   - Removed statistical outliers using IQR method

2. **Exploratory Data Analysis**
   - Price distribution analysis
   - Correlation heatmaps
   - Payment method analysis
   - Feature relationship visualization

3. **Feature Engineering**
   - Price per square meter
   - Total rooms (bedrooms + bathrooms)
   - Bedroom to bathroom ratio
   - Property size categories
   - One-hot encoded categorical features

4. **Model Development**
   - Tested 8 different algorithms
   - Compared 3 feature sets (Basic, Enhanced, All)
   - Hyperparameter tuning for best models
   - Cross-validation and performance metrics

## 🏆 Model Comparison Results

| Model | Feature Set | R² Score | MAE (EGP) | RMSE (EGP) |
|-------|-------------|----------|-----------|------------|
| **Gradient Boosting** | **All** | **0.3040** | **4,710,663** | **6,200,021** |
| Gradient Boosting | Enhanced | 0.2949 | 4,759,843 | 6,240,330 |
| Random Forest | All | 0.2638 | 4,753,302 | 6,376,531 |
| Linear Regression | All | 0.2669 | 4,870,379 | 6,362,939 |

## 💡 Key Insights

### Feature Importance (Gradient Boosting Model)
1. **Size (53.4%)**: Property size is the most critical pricing factor
2. **Bathrooms (18.2%)**: Number of bathrooms significantly impacts price
3. **Total Rooms (17.5%)**: Combined room count matters
4. **Bed-Bath Ratio (8.8%)**: Room balance affects pricing
5. **Bedrooms (1.8%)**: Direct bedroom count has minimal impact

### Business Insights
- Property size dominates pricing decisions
- Bathrooms add more value than additional bedrooms
- Feature engineering significantly improves model performance
- Tree-based models outperform linear regression for this dataset

## 🛠️ Technologies Used

- **Python 3.13**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Environment**: Jupyter Notebook, Virtual Environment

## 📁 Project Structure

```
real_estate_prediction/
├── dataanalysis.ipynb              # Main analysis notebook
├── egypt_real_estate_listings.csv  # Dataset
├── archive.zip                     # Original data archive
└── README.md                       # This documentation
```

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/ikapil06/Natural-language-processing-NLP-.git
   cd Natural-language-processing-NLP-/real_estate_prediction
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn jupyter
   ```

4. **Run the analysis**
   ```bash
   jupyter notebook dataanalysis.ipynb
   ```

## 📈 Model Performance Metrics

- **R² Score**: 0.3040 (explains 30.4% of price variance)
- **Mean Absolute Error**: 4,710,663 EGP
- **Root Mean Square Error**: 6,200,021 EGP
- **Data Retention**: 85.1% after cleaning
- **Feature Count**: 9 engineered features

## 🎯 Model Development Journey

1. **Original Linear Regression**: 0.2532 R²
2. **Data Cleaning**: Maintained 85.1% of data, removed outliers
3. **Feature Engineering**: Added 5 new features
4. **Multiple Model Testing**: Tested 8 different algorithms
5. **Hyperparameter Tuning**: Fine-tuned best performing model
6. **Final Model**: **+20.1% improvement** (0.3040 R²)

## 🔮 Future Improvements

- Include location-based features (neighborhood analysis)
- Add temporal features (market trends, seasonality)
- Experiment with deep learning models
- Incorporate external economic indicators
- Implement ensemble methods combining multiple models

## 📝 Results Summary

The final Gradient Boosting model successfully predicts Egyptian real estate prices with 30.4% accuracy, explaining nearly one-third of price variations. This represents a significant improvement over basic linear regression and provides valuable insights for:

- Property valuation
- Market analysis
- Investment decisions
- Price estimation tools

## 👨‍💻 Author

**Kapil**
- GitHub: [@ikapil06](https://github.com/ikapil06)

---

*This project demonstrates a comprehensive machine learning workflow from data cleaning to model deployment for real estate price prediction in the Egyptian market.*