# 🏠 Egypt House Price Prediction

An interactive Streamlit app that forecasts residential property prices in Egypt using machine learning. Users can input specific property features and receive a real-time price prediction based on a trained model.

> Built with 💡 by [Islam Abd Aljawad](https://www.linkedin.com/in/islamabdaljawad/) , [Mohamed Adel](https://www.linkedin.com/in/mohamed-adel-14885b248?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) , [Rana Ashraf](https://www.linkedin.com/in/rana-ashraf-349a52198/) , [Ahmed Abdelbaset](https://www.linkedin.com/in/ahmed-samy-abdelbaset-40060620a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) , [Habiba Kandil](https://www.linkedin.com/in/islamabdaljawad/)

---

## 📸 App Preview

![App Screenshot](https://github.com/islamabdaljawad/forecast-egypt-property-values/blob/main/Images/streamlit%20web.JPG)  

---

## 🔍 Project Overview

This project aims to help users estimate property values in Egypt using historical real estate data. The app:

- Analyzes and cleans real listing data
- Trains and compares multiple ML models
- Uses the best-performing model for predictions
- Provides a clean UI built with Streamlit

---

## 🧹 Data Cleaning

### 🧩 Problems in the Raw Data

- Inconsistent and messy **city, compound, and delivery** names  
- Missing or null values in key fields like `price`, `area`, `delivery_date`  
- Outliers in `price` (too low/high) and `area` (too small/large)  
- Duplicate or irrelevant columns  
- Text values mixed in Arabic and English, with typos

---

### ✅ Solutions Applied

- Standardized city and compound names (e.g., "6th of October" → "6 October")  
- Removed or imputed rows with missing or irrelevant values  
- Filtered extreme values using domain knowledge  
- Converted `delivery_date` to "delivery time in months"  
- Grouped low-frequency compounds under `"Unknown"`  
- Cleaned up all category label inconsistencies

---

### 🔧 Data Cleaning Process

1. **Drop irrelevant columns** (e.g., agent info, IDs, links)  
2. **Handle missing data**:
   - Removed rows with missing target (`price`)
   - Dropped features with over 30% missing values
3. **Fix categorical labels** (standard casing, language unification)  
4. **Create new features**:
   - `delivery_time_in_months` from current date and delivery date  
   - Binary `furnished` feature
5. **Outlier filtering**:
   - `price` between 100,000 and 50,000,000 EGP  
   - `area` between 20 and 1000 sqm  
6. **Exported cleaned dataset** for modeling and app integration

---

## 🤖 Model Selection & Evaluation

Several machine learning regression models were trained and tested to predict house prices.

### 📊 Models and Their Results

| Model                      | RMSE  | R² Score |
|---------------------------|-------|----------|
| **Random Forest**             | 0.70  | 0.65     |
| **Random Forest with PCA**    | 0.81  | 0.53     |
| **Support Vector Regression** | 0.83  | 0.50     |
| **XGBoost**                   | 0.70  | 0.65     |
| **LightGBM**                  | 0.70  | 0.65     |
| **Linear Regression**         | 0.82  | 0.52     |
| **Neural Network**            | 0.74  | 0.61     |
| **Other Neural Network**      | 0.74  | 0.60     |
| **Stacked Models**            | 0.70  | 0.65     |

---


### 🏆 Best Performing Model

- LightGBM Regressor  
- Chosen for its *small model size* and *high accuracy*  
    Efficient for deployment and fast inference
Achieved strong results:

  - R² Score: 0.65
  - RMSE: 700,000 EGP

---

## 🛠 Features of the App

- Predicts prices for multiple property types (Apartment, Villa, Duplex, etc.)  
- Supports inputs for:
  - Property type, city, area, furnishing status  
  - Bedrooms, bathrooms, delivery term and date  
  - Compound (or unknown), payment method  
- User-friendly real-time predictions in Streamlit

---

## 🗂 Project Structure

```
.
├── main.py                # Streamlit app
├── src/
│   ├── utils.py           # Prediction logic
│   ├── filter.py          # Input filters and options
│   └── model.pkl          # Trained ML model
├── notebooks/
│   ├── egypt-houses-market-analysis.ipynb
│   └── forecasting-house-prices-in-egypt.ipynb
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## 🧪 Notebooks

- `egypt-houses-market-analysis.ipynb`: Exploratory analysis, cleaning strategy  
- `forecasting-house-prices-in-egypt.ipynb`: Model training and evaluation  

---

## 🚀 How to Run the App Locally

1. **Clone the repo**  
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the app**  
   ```bash
   streamlit run main.py
   ```

---

## 🔮 Future Improvements

- Deploy to Streamlit Cloud  
- Add map visualizations for city/compound pricing  
- Use SHAP for explainability  
- Allow CSV upload for batch predictions

---

## 👤 Author

**Islam Abd Aljawad Ahmed**  
[LinkedIn](https://www.linkedin.com/in/islamabdaljawad) • [GitHub](https://github.com/islamabdaljawad)  • [Email](mailto:islamzabdzallah@gmail.com)

**Mohammed Adel Mohammed**  
 [LinkedIn](https://www.linkedin.com/in/mohamed-adel-14885b248?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
• [GitHub](https://github.com/MohAdel13)
• [Email](mohammedofficial1311@gmail.com)

**Rana Ashraf Mahmoud**  
[LinkedIn](https://www.linkedin.com/in/rana-ashraf-349a52198/) • [GitHub](https://github.com/Ranaashraff8)  • [Email](mailto:rana.ashraf.1197@gmail.com)

**Ahmed samy Abdelbaset**  
[LinkedIn](https://www.linkedin.com/in/ahmed-samy-abdelbaset-40060620a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) • [GitHub](https://github.com/ahmedabdelbast) • [Email](mailto:a7medabdelbast@gmail.com)



---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project helpful, feel free to give it a ⭐ on GitHub and share it with others!

---

Let me know if you'd like a badge section, a hosted version added, or an interactive table in the README!
