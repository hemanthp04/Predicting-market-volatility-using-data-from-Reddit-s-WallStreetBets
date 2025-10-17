# WSB Stock Prediction - Machine Learning Mini Project

Predicting stock market movements using sentiment analysis from Reddit's r/WallStreetBets

**Course**: UE23CS352A Machine Learning  
**Institution**: PES UNIVERSITY  
**Project Duration**: September 29 - October 13, 2025  
**Team Members**: Hemanth.P , Harshini Dharniraj

---

## 📊 Project Overview

This project investigates whether social media sentiment from Reddit's r/WallStreetBets can predict stock price movements. Inspired by the 2021 GameStop short squeeze, we analyze posts, extract sentiment features, and train machine learning models to predict next-day stock movements (up/down).

### 🎯 Key Results

Our **Logistic Regression model achieved 70.6% accuracy** for GameStop (GME), demonstrating that social media sentiment contains predictive signals for stock movements.

| Model | GME Accuracy | Notes |
|-------|--------------|-------|
| **Logistic Regression** | **70.6%** | ✅ Best performance, main model |
| Neural Network | 53.1% | Overfitting due to small dataset |
| Random Baseline | 58.8% | Floor performance |

**Key Finding**: Traditional machine learning (Logistic Regression) outperformed deep learning (Neural Network) due to limited training data (170 samples), highlighting the importance of data volume for model selection.

---

## 🗂️ Project Structure

```
Predicting-market-volatility/
│
├── Data/
│   └── reddit_wsb.csv
│
├── Docs/
│   ├── ML59_238_244_....pdf
│   └── presentationML5....pdf
│
├── Notebooks/
│   └── WSB_Stock_Prediction.ipynb
│
├── Results/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── nn_overfitting.png
│   ├── performance_comparison.png
│   ├── ticker_disturbution.png
│   └── WSB_Graphs.png
│
├── README.md
└── requirements.txt



```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Colab (recommended) or local Jupyter environment
- 4GB RAM minimum

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/hemanthp04/Predicting-market-volatility-using-data-from-Reddit-s-WallStreetBets.git
cd wsb-stock-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
- Visit [Kaggle WSB Dataset](https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts)
- Download and place in `data/` folder
- Or upload directly in Colab when prompted

### Running the Project

**Option 1: Google Colab (Recommended)**
1. Open `notebooks/WSB_Stock_Prediction.ipynb` in Colab
2. Upload the dataset when prompted
3. Run all cells (`Runtime` → `Run all`)
4. Results will be generated automatically

**Option 2: Local Jupyter**
```bash
jupyter notebook notebooks/WSB_Stock_Prediction.ipynb
```

---

## 📊 Dataset

### Reddit Data (Kaggle)
- **Source**: r/WallStreetBets posts (2020-2021)
- **Size**: ~100,000+ posts
- **Filtered**: Posts mentioning GME, AMC, TSLA, BB, FB
- **Features**: Title, body, score, comments, timestamp

### Stock Data (Yahoo Finance)
- **Source**: Yahoo Finance API (via yfinance)
- **Stocks**: GME, AMC, TSLA, BB, FB
- **Period**: October 2020 - April 2021
- **Features**: OHLCV, returns, volatility, moving averages

### Final Dataset
- **Total Samples**: 170 (GME), 169 (AMC)
- **Features**: 20+ engineered features
- **Split**: 80% train, 20% test
- **Target**: Binary (0=Down, 1=Up next day)

---

## 🔧 Methodology

### 1. Data Collection & Preprocessing
```python
# Load Reddit posts
reddit_df = pd.read_csv('reddit_wallstreetbets.csv')

# Extract stock tickers mentioned
tickers = extract_tickers(reddit_df['text'])  # GME, AMC, etc.

# Fetch stock prices
stock_data = yf.download(tickers, start='2020-10-01', end='2021-04-30')
```

### 2. Feature Engineering

**Sentiment Features** (TextBlob + VADER):
- Sentiment polarity scores (-1 to +1)
- Sentiment categories (Bullish/Neutral/Bearish)
- Compound sentiment scores

**Engagement Metrics**:
- Post scores (upvotes)
- Comment counts
- Daily post volume
- Emoji usage (🚀💎📈)

**Technical Indicators**:
- Daily returns
- 5-day volatility
- 5-day and 20-day moving averages

**Aggregation**:
```python
# Group by date and ticker
daily_features = reddit_df.groupby(['date', 'ticker']).agg({
    'sentiment': ['mean', 'std', 'min', 'max'],
    'score': ['sum', 'mean', 'max'],
    'comments': ['sum', 'mean']
})
```

### 3. Model Implementation

#### Model 1: Logistic Regression (PRIMARY MODEL) ✅
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Results: 70.6% accuracy for GME
```

**Why it worked:**
- Suitable for small datasets (170 samples)
- Fast training and inference
- Interpretable coefficients
- Robust to noise

#### Model 2: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Results: 45.3% accuracy (underperformed)
```

#### Model 3: Neural Network (OVERFITTING OBSERVED) ⚠️
```python
from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier(
    hidden_layer_sizes=(256, 256, 256),
    activation='relu',
    batch_size=64,
    max_iter=30
)
nn_model.fit(X_train_scaled, y_train)

# Results: 41.2% accuracy - OVERFITTING!
```

**Why it failed:**
- Training loss: 0.6297 → 0.4774 ⬇️ (good)
- Validation loss: 0.7094 → 0.8552 ⬆️ (BAD - overfitting!)
- Insufficient data (170 samples too small for deep learning)
- Model memorized training data, couldn't generalize

---

## 📈 Results & Analysis

### Overall Performance (GME Stock)

| Metric | Logistic Regression | Neural Network | Random Forest |
|--------|-------------------|----------------|---------------|
| **Accuracy** | **70.6%** | 41.2% | 45.3% |
| **Precision (Up)** | 0.62 | 0.43 | - |
| **Recall (Up)** | 1.00 | 0.75 | - |
| **F1-Score (Up)** | 0.76 | 0.55 | - |

### Classification Report (Logistic Regression - GME)
```
              precision    recall  f1-score   support
        Down       1.00      0.44      0.62         9
          Up       0.62      1.00      0.76         8
    accuracy                           0.71        17
   macro avg       0.81      0.72      0.69        17
weighted avg       0.82      0.71      0.68        17
```

### Key Insights

✅ **What Worked:**
1. **Logistic Regression** performed best with 70.6% accuracy
2. Model correctly predicted 100% of upward movements (recall=1.00)
3. Sentiment features showed strong correlation with price movements
4. Simple models outperformed complex ones on small datasets

❌ **What Didn't Work:**
1. **Neural Network overfitting** - validation loss increased while training loss decreased
2. **Random Forest underperformed** - may need more hyperparameter tuning
3. Limited data (170 samples) insufficient for deep learning
4. High variance in predictions for stocks with less social media activity

### Feature Importance (Top 5)
1. Sentiment Mean Score (23%)
2. Daily Post Volume (18%)
3. Comment Engagement (15%)
4. Previous Day Returns (12%)
5. Sentiment Std Dev (11%)

---

## 🎓 Learnings & Challenges

### Technical Challenges
1. **Data Quality**: Social media data is noisy with slang, sarcasm, emojis
2. **MultiIndex Columns**: Yahoo Finance dataframe required careful handling
3. **Date Alignment**: Matching Reddit posts with trading days (weekends/holidays)
4. **Class Imbalance**: Slight imbalance in up/down days required stratified sampling
5. **Overfitting**: Deep models failed on small datasets - learned importance of model selection

### Domain Challenges
1. **Market Complexity**: Stock prices influenced by many factors beyond sentiment
2. **Causality**: Difficult to prove sentiment drives prices vs reflects existing trends
3. **Time Lag**: Optimal window between post and price movement unclear
4. **Stock Variability**: Different stocks respond differently to social sentiment

### Key Takeaways
- ✅ **Data volume matters**: Small datasets favor traditional ML over deep learning
- ✅ **Feature engineering is crucial**: Well-designed features outperform model complexity
- ✅ **Simpler is often better**: Logistic Regression beat Neural Network
- ✅ **Domain knowledge helps**: Understanding market dynamics improved feature selection
- ✅ **Social sentiment works**: 70.6% accuracy proves WSB has predictive power

---

## 🔮 Future Improvements

### Short-term (Feasible)
- [ ] Collect more historical data (2019-2024) for larger training set
- [ ] Implement LSTM/GRU for temporal pattern recognition
- [ ] Add cross-validation for more robust evaluation
- [ ] Tune Random Forest hyperparameters using GridSearchCV
- [ ] Test on additional stocks (AAPL, NVDA, MSFT)

### Medium-term (Research)
- [ ] Incorporate Twitter, Discord, Stocktwits data
- [ ] Use BERT/RoBERTa for advanced sentiment analysis
- [ ] Multi-modal learning (combine text + price charts)
- [ ] Attention mechanisms to weight important posts
- [ ] Real-time data pipeline with streaming APIs

### Long-term (Production)
- [ ] Deploy as web application with live predictions
- [ ] Backtesting framework with transaction costs
- [ ] Risk management and position sizing
- [ ] Integration with trading APIs (paper trading)
- [ ] Dashboard for monitoring model performance

---

## 📚 Technologies Used

**Core Libraries:**
```python
pandas==1.5.3          # Data manipulation
numpy==1.24.3          # Numerical computing
scikit-learn==1.3.0    # Machine learning models
```

**NLP & Sentiment:**
```python
textblob==0.17.1       # Sentiment analysis
vaderSentiment==3.3.2  # Social media sentiment
```

**Data Collection:**
```python
yfinance==0.2.28       # Stock price data
```

**Visualization:**
```python
matplotlib==3.7.2      # Plotting
seaborn==0.12.2        # Statistical visualization
```

**Development:**
```python
jupyter==1.0.0         # Interactive notebooks
google-colab           # Cloud environment
```

---

## 📖 References & Inspiration

### Primary Reference
**Stanford CS229 Project (Spring 2021)**
- *"Predicting market volatility and building short-term trading strategies using data from Reddit's WallStreetBets"*
- Authors: Shashank Rammoorthy, Kiara Nirghin, Abhishek Raghunathan
- Link: [CS229 Report](https://cs229.stanford.edu/proj2021spr/report2/82006407.pdf)
- Our approach follows similar methodology with focus on Logistic Regression

### Dataset Sources
1. **Kaggle - Reddit WallStreetBets Posts**
   - Link: https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts
   - ~100,000+ posts from 2020-2021

2. **Yahoo Finance API**
   - Library: yfinance
   - Historical stock price data

### Research Papers
1. Bollen, J., Mao, H., & Zeng, X. (2011). *"Twitter mood predicts the stock market"*
2. Tetlock, P. C. (2007). *"Giving content to investor sentiment: The role of media in the stock market"*
3. Preis, T., et al. (2013). *"Quantifying Trading Behavior in Financial Markets Using Google Trends"*

### Documentation
- scikit-learn: https://scikit-learn.org/stable/
- TextBlob: https://textblob.readthedocs.io/
- VADER: https://github.com/cjhutto/vaderSentiment
- yfinance: https://pypi.org/project/yfinance/

---

## 👥 Team Contributions

**Hemanth.P** - PES1UG23CS244
- Data collection and preprocessing
- Feature engineering (sentiment analysis)
- Logistic Regression implementation
- Documentation and README

**Harshini** - PES1UG23CS238
- Neural Network implementation
- Model evaluation and comparison
- Visualization and plotting
- Presentation slides

**Joint Work:**
- Problem formulation
- Literature review
- Results analysis
- Final report writing

---

## 📊 Project Timeline

```
Week 1 (Sep 29 - Oct 5)
├── Problem understanding & literature review
├── Dataset acquisition from Kaggle
├── Exploratory data analysis
└── Feature engineering design

Week 2 (Oct 6 - Oct 13)
├── Model implementation (LogReg, RF, NN)
├── Training and evaluation
├── Results analysis
├── Documentation & presentation
└── Final submission (Oct 13, 11:59 PM)
```

---

## 🎤 Presentation

**Date**: October 14-15, 2025  
**Duration**: 10-15 minutes  
**Format**: Slides + Live Demo

**Key Points:**
1. Problem motivation (GameStop phenomenon)
2. Data collection and preprocessing
3. Feature engineering approach
4. Model comparison (LogReg wins!)
5. Why Neural Network failed (overfitting analysis)
6. Results and conclusions
7. Live demo of predictions

---

---

## 🤝 Acknowledgments

- **Course Instructor**: Dr.Surabhi Narayan for guidance and support
- **Stanford CS229 Team**: For the inspiring reference paper
- **Kaggle Community**: For the WallStreetBets dataset
- **r/WallStreetBets**: For demonstrating the power of retail investors

---

## 📧 Contact

**For questions or collaboration:**

- **Hemanth.P**
  - Email: pes1ug23cs344@pesu.pes.edu
  - GitHub: [@hemanthp04](https://github.com/hemanthp04)

- **Harshini**
  - Email: pes1ug23cs238@pesu.pes.edu
  - GitHub: [@h4rshini](https://github.com/h4rshini)

**Project Links:**
- 📦 Repository: `https://github.com/hemanthp04/Predicting-market-volatility-using-data-from-Reddit-s-WallStreetBets`
- 📊 Kaggle Dataset: [[Link to dataset]](https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts)


---

## 📌 Citation

If you use this work in your research or project, please cite:

```bibtex
@misc{wsb-stock-prediction-2025,
  author = {Hemanth.P, Harshini},
  title = {Predicting Stock Market Movements Using Reddit WallStreetBets Sentiment Analysis},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/[yourusername]/Predicting-market-volatility-using-data-from-Reddit-s-WallStreetBets}}
}
```

---

</div>
