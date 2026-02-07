# Understanding-Probabilistic-Distribution-Prediction-over-Time-Series-using-Deep-Learning
## Goal
To achieve good predictive performance, but also to understand what is fundamentally predictable and what is not from the available data.

## Problem Statement
Given historical household electricity usage and related features, predict the next-step Global Active Power consumption.

## Dataset

Dataset Source:- [Link](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
The dataset contains household-level electricity measurements with the following features:

### Power-related features

Global_active_power
Global_intensity
Sub_metering_1
Sub_metering_2
Sub_metering_3

### Time-based features

year
quarter
month
day
weekday

## Data Preprocessing
Conversion of numeric features to appropriate data types

<img width="322" height="187" alt="image" src="https://github.com/user-attachments/assets/4ce529e2-cc6b-4080-9934-d7deaa14552f" />

Handling missing values
df = df.dropna(subset=cols_to_process)

Log transformation of the target variable to stabilize variance
dataset = np.log1p(dataset)

Feature scaling using MinMaxScaler

Temporal train–test split in 80:20

Sliding window generation using tf.data.Dataset

## Model Architecture

A multivariate LSTM is used as a deterministic baseline model.

### Architecture:

Input: past L timesteps × all features

### LSTM layer

Dropout (regularization)

Dense output layer (single-step prediction)

### Training setup:

Loss: Mean Squared Error (MSE)

Optimizer: Adam

Early stopping with best weight restoration

No shuffling (temporal order preserved)

## Evaluation Strategy

### Model performance is evaluated using:

Quantitative metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

### Qualitative diagnostics

Actual vs predicted time-series plots

<img width="808" height="408" alt="image" src="https://github.com/user-attachments/assets/29b47fa8-c9ec-4842-93f4-112002c7d1e6" />

Residual distribution (histogram)

<img width="571" height="416" alt="image" src="https://github.com/user-attachments/assets/fd418afb-50b2-47a9-a9dc-34af74d80ecb" />

These diagnostics help assess how and why the model fails, not just how often.

## Results

The model accurately captures average consumption levels and smooth temporal dynamics.

Performance is strong on typical behavior and stable regimes.

Sudden spikes are often predicted with delay.

This behavior is consistent across both training and test sets.

## Key Insights
1. Predictability is limited by information

Sudden spikes are caused by unobserved external events (e.g., appliance switching).
Without leading indicators, these events are fundamentally unpredictable.

2. The model learns the conditional mean

Training with MSE leads the model to predict the expected next value, which results in:

Smoothing of rare spikes

Delayed reaction instead of anticipation

This is optimal behavior, not a modeling bug.

3. Residuals reveal uncertainty structure

Residual analysis shows:

Strong concentration near zero (model usually correct)

Heavy tails (rare but large errors)

No strong bias (symmetric error distribution)

This indicates irreducible uncertainty, not underfitting.

## Limitations

No access to causal event features (appliance states, occupancy, etc.)

Deterministic predictions only (mean, not uncertainty)

Single-step forecasting (no long-horizon evaluation)

## Future Work

Potential extensions include:

Probabilistic forecasting (predicting uncertainty, not just mean)

Quantile regression or distributional outputs

Incorporating event-aware or behavioral features

Multi-step forecasting analysis

Removing redundent featues as through correlation redundency over features have been observed in terms of hypothesis.

<img width="643" height="543" alt="image" src="https://github.com/user-attachments/assets/b08ab30c-c15a-4887-a25b-adb47e041149" />


## Technologies Used

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

scikit-learn

License

This project is released under the MIT License.

## Final Note

This project emphasizes understanding model behavior and limitations, not just optimizing metrics.
It serves as a baseline for exploring uncertainty-aware forecasting in real-world time-series data.
