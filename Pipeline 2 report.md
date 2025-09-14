Research Report: Development of a Sequential Deep Learning Model (Pipeline 2)
Date: September 12, 2025
Author: Ayan, Gemini Quantitative Analysis
Status: Final

Executive Summary
This report details the systematic development and testing of "Pipeline 2: The Sequential Specialist," a deep learning model designed for next-day (t+1) stock price forecasting. The primary objective was to test the hypothesis that a sequential model architecture could find predictive patterns in time-series data that were not captured by the feature-based champion model from Pipeline 1 (Directional Accuracy: 55.59%).

The research explored multiple state-of-the-art architectures, including Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), and a hybrid Convolutional Neural Network (CNN) + GRU model.

Despite rigorous testing and architectural evolution, all versions of Pipeline 2 failed to achieve the performance of the simpler, faster Pipeline 1 model. The final GRU model achieved a peak Directional Accuracy of only 53.09% with a prohibitive runtime of nearly 5 hours.

The research concluded that for this specific problem, the sequential deep learning approach was a suboptimal strategy, likely due to the high noise levels in financial data and the computational expense of the walk-forward validation framework.

1. Introduction & Objective
After the successful development of a feature-based model in Pipeline 1, the Pipeline 2 research track was initiated to explore a fundamentally different approach. The hypothesis was that a model capable of learning long-range temporal dependencies directly from sequences of market data could potentially find a more powerful signal. The goal was to build a sequential model that could outperform the Pipeline 1 champion's directional accuracy of 55.59%.

2. Methodology & Infrastructure
The research followed a structured process, beginning with a classic LSTM and evolving to more modern architectures based on performance. The same rigorous walk-forward validation framework was used.

Modeling: tensorflow, keras

Data Handling: pandas, numpy, scikit-learn

MLOps: DVC (Data Version Control), MLflow (Experiment Tracking)

3. Data Sources
The sequential models were trained on a focused set of features designed to provide a "snapshot" of the market at each timestep.

Primary Market Data: Daily OHLCV data for the target stock.

Local Market Index: S&P 500 (^GSPC)

Volatility Index: CBOE VIX (^VIX)

Derived Features: Daily returns, index returns, volatility changes, RSI, and Bollinger Band Width.

4. Pipeline Evolution & Results
The pipeline evolved through multiple architectures in an attempt to find a predictive edge.

Version

Architecture

Runtime

RMSE

MAE

Dir. Acc.

Analysis & Learning

V1

LSTM (Minimal Features): A standard stacked LSTM with a basic feature set.

~3.5h

2.71

2.05

50.10%

FAILURE. The model was extremely slow and its accuracy was no better than a coin flip. The minimalist feature set was insufficient.

V2

GRU (Richer Features): Evolved to a faster GRU architecture and a richer feature set including technical indicators.

~4.5h

2.66

2.01

50.10%

CONCLUSIVE FAILURE. Despite being an excellent magnitude predictor (best RMSE/MAE), the model's directional accuracy showed no improvement. The core hypothesis was invalidated.

V3

Hybrid CNN-GRU: A more complex model designed to be faster and more accurate by using CNNs to pre-process the sequence for the GRU.

~2.5h

2.67

2.01

50.30%

FAILURE (OVERFITTING). The added complexity made the model worse, indicating it was likely overfitting to noise in the training data.

5. Final Model Specification (The GRU Model)
The most successful version of Pipeline 2 was the GRU architecture.

Architecture: A stacked Gated Recurrent Unit (GRU) network.

Core Features: A sequence of daily feature vectors including returns, index changes, volatility changes, RSI, and Bollinger Band Width.

Target: Next-day price return.

Training: Optimized walk-forward validation with periodic retraining.

Final Performance:

RMSE: 2.6649

MAE: 2.0138

Directional Accuracy: 50.10%

6. Conclusion
The Pipeline 2 research project was a valuable and conclusive exercise. It successfully demonstrated that for this specific forecasting problem, sequential deep learning models, despite their sophistication, are not the optimal tool. They are computationally expensive and highly susceptible to overfitting on noisy financial data.

The failure of Pipeline 2 to outperform Pipeline 1 provides strong, data-driven evidence that the feature-based, gradient boosting approach is the superior strategy. This concludes the research into sequential models for this project.