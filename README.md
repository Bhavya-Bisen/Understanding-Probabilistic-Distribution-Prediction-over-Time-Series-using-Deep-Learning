# Understanding-Probabilistic-Distribution-Prediction-over-Time-Series-using-Deep-Learning
## Goal
To achieve good predictive performance, but also to understand what is fundamentally predictable and what is not from the available data.

## Problem Statement
Given historical household electricity usage and related features, predict the next-step Global Active Power consumption.

## Dataset

Dataset Source:- [Link](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)</br>
The dataset contains household-level electricity measurements with the following features:

### Power-related features

Global_active_power</br>
Global_intensity</br>
Sub_metering_1</br>
Sub_metering_2</br>
Sub_metering_3</br>

### Time-based features

year</br>
quarter</br>
month</br>
day</br>
weekday</br>

## Data Preprocessing
Conversion of numeric features to appropriate data types

<img width="322" height="187" alt="image" src="https://github.com/user-attachments/assets/4ce529e2-cc6b-4080-9934-d7deaa14552f" />

Handling missing values</br>
``` df = df.dropna(subset=cols_to_process)</br> ```

Log transformation of the target variable to stabilize variance</br>
```dataset = np.log1p(dataset)</br>```

Feature scaling using MinMaxScaler</br>

Temporal train–test split in 80:20</br>

Sliding window generation using</br>
```tf.data.Dataset</br>```

## Model Architecture

A multivariate LSTM is used as a deterministic baseline model.

### Architecture:

Input: past L timesteps × all features

### LSTM layer

Dropout (regularization)</br>

Dense output layer (single-step prediction)</br>

### Training setup:

Loss: Mean Squared Error (MSE)</br>

Optimizer: Adam</br>

Early stopping with best weight restoration</br>

No shuffling (temporal order preserved)</br>

## Evaluation Strategy

### Model performance is evaluated using:

Quantitative metrics</br>

Mean Absolute Error (MAE)</br>

Root Mean Squared Error (RMSE)</br>

### Qualitative diagnostics

Actual vs predicted time-series plots</br>

<img width="808" height="408" alt="image" src="https://github.com/user-attachments/assets/29b47fa8-c9ec-4842-93f4-112002c7d1e6" />

Residual distribution (histogram)</br>

<img width="571" height="416" alt="image" src="https://github.com/user-attachments/assets/fd418afb-50b2-47a9-a9dc-34af74d80ecb" />

These diagnostics help assess how and why the model fails, not just how often.</br>

## Results

The model accurately captures average consumption levels and smooth temporal dynamics.</br>

Performance is strong on typical behavior and stable regimes.</br>

Sudden spikes are often predicted with delay.</br>

This behavior is consistent across both training and test sets.</br>

## Key Insights
1. Predictability is limited by information</br>

Sudden spikes are caused by unobserved external events (e.g., appliance switching).</br>
Without leading indicators, these events are fundamentally unpredictable.</br>

2. The model learns the conditional mean</br>

Training with MSE leads the model to predict the expected next value, which results in:</br>

Smoothing of rare spikes</br>

Delayed reaction instead of anticipation</br>

This is optimal behavior, not a modeling bug.</br>

3. Residuals reveal uncertainty structure</br>

Residual analysis shows:</br>

Strong concentration near zero (model usually correct)</br>

Heavy tails (rare but large errors)</br>

No strong bias (symmetric error distribution)</br>

This indicates irreducible uncertainty, not underfitting.</br>

## Limitations

No access to causal event features (appliance states, occupancy, etc.)</br>

Deterministic predictions only (mean, not uncertainty)</br>

Single-step forecasting (no long-horizon evaluation)</br>

## Future Work

Potential extensions include:</br>

Probabilistic forecasting (predicting uncertainty, not just mean)</br>

Quantile regression or distributional outputs</br>

Incorporating event-aware or behavioral features</br>

Multi-step forecasting analysis</br>

Removing redundent featues as through correlation redundency over features have been observed in terms of hypothesis.</br>

<img width="643" height="543" alt="image" src="https://github.com/user-attachments/assets/b08ab30c-c15a-4887-a25b-adb47e041149" />


## Technologies Used

Python</br>

TensorFlow / Keras</br>

NumPy, Pandas</br>

Matplotlib, Seaborn</br>

scikit-learn</br>

License</br>

This project is released under the MIT License.</br>

## Final Note

This project emphasizes understanding model behavior and limitations, not just optimizing metrics.</br>
It serves as a baseline for exploring uncertainty-aware forecasting in real-world time-series data.</br>
