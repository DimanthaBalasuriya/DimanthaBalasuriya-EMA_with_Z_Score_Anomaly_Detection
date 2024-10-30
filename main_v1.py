"""

    Explanation of the Algorithm and approach

   The goal of this algorithm is to spot unusual patterns, called anomalies, in a stream of data that typically has a regular trend or pattern.
   Think of it like noticing if a sudden spike or dip appears in what is usually a steady line of information—like noticing if someone starts
   speeding up and slowing down randomly on a normally smooth road.

    I first build a pattern in the data that includes a repeating cycle (like seasons) and a slow, steady increase (trend), with some random variations (noise).
    This mix simulates real-life data that typically has these features, such as temperatures over the year or website visits that grow slowly with seasonal peaks.

    To simulate unexpected events, I inject random "spikes" at certain points. These are similar to sudden, unusual events that can sometimes occur, like a big
    jump in sales during a sale day.

    To understand the normal trend, the algorithm calculates an "Exponential Moving Average" (EMA), which follows the general trend of the data and updates as new
    data points come in. EMA is like a flexible line that tracks the overall direction of the data without getting thrown off by each little wiggle

    Next, the algorithm calculates a measure of how much each point strays from the EMA, using something called the "Z-score." A Z-score tells us how far a data point
    is from the average trend line. If a point has a high Z-score, it means it’s unusually far from what the trend predicts.

    If the Z-score of a point is beyond a set threshold (e.g., 2.5), it’s flagged as an anomaly. This threshold means that the point is far enough from the trend that
    it’s worth a closer look, like noticing if your heartbeat jumps during a workout.

    The algorithm then outputs a list of points that are considered anomalies—those that don’t fit the usual pattern. These flagged points may indicate something
    noteworthy, like errors or significant changes in the data’s underlying process.

    By combining the trend (EMA) with variability (Z-score), this algorithm identifies anomalies based on how far points deviate from a baseline pattern, even if
    the overall pattern changes gradually. It’s an efficient way to spot unusual data in settings where new data comes in continuously, such as financial transactions
    or sensor readings. This method ensures that only truly unusual values are flagged, without getting distracted by regular fluctuations or slow changes.

    Here are the issues in this solution

    Recalculating for each new point in large datasets is slow. Consider using more efficient libraries or an incremental variance method.
    Repeated EMA calculations can add overhead. Using simpler math libraries or hardware acceleration may help.
    Frequent probability calculations may slow down large datasets. Batch injecting anomalies could optimize performance.
    Chained calculations in pandas add latency. Perform calculations outside the DataFrame and add results back afterward.
    Static plotting isn’t suitable for real-time streams. Use interactive libraries like Plotly or a lightweight dashboard.
    Storing large, continuous data in the DataFrame uses excessive memory. Use a sliding window to limit data in memory to recent points only.

    There are many algorithms can use to diagnose anomalies in data stream.

    Incremental or Online Algorithms
    Statistical Process Control (SPC)
    Machine Learning Models
    Approximate Nearest Neighbor (ANN) Search
    Fourier or Wavelet Transforms
    Sliding Window with Summaries

    Most of these solutions are need to use more memory, more cpu than the implemented solution, but these solution can be accurate than the implemented solution in different
    scenarios.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for the data stream
np.random.seed(100)  # Set seed for reproducibility of random values
data_points = 1000   # Total number of points in the data stream
season_ratio = 50  # Period for the sine wave seasonal component
noise_ratio = 0.5   # Noise level to simulate random variability
anomaly_ratio = 0.05 # Probability of adding anomalies to data points

# Validate data stream parameters to avoid potential issues with data generation
if data_points <= 0:
    raise ValueError("data_points must be a positive integer.")
if season_ratio <= 0:
    raise ValueError("season_ratio must be a positive integer.")
if not (0 <= noise_ratio <= 1):
    raise ValueError("noise_ratio should be between 0 and 1.")
if not (0 <= anomaly_ratio <= 1):
    raise ValueError("anomaly_ratio should be between 0 and 1.")

# Generate the time series data stream with seasonal, trend, and noise components
try:
    time = np.arange(data_points)  # Time points from 0 to data_points - 1
    seasonal_component = np.sin(2 * np.pi * time / season_ratio)  # Create a repeating sine wave pattern
    trend_component = 0.005 * time  # Linear trend that gradually increases
    noise_component = np.random.normal(0, noise_ratio, data_points)  # Random noise around zero
    data_stream = seasonal_component + trend_component + noise_component  # Final data stream with trend, seasonality, and noise
except Exception as e:
    raise RuntimeError(f"Error generating data stream: {e}")

# Inject anomalies into the data stream to simulate unusual spikes
try:
    anomalies = np.random.rand(data_points) < anomaly_ratio  # Identify random points for anomalies based on probability
    data_stream[anomalies] += np.random.normal(5, 1, anomalies.sum())  # Add random spikes to these points
except Exception as e:
    raise RuntimeError(f"Error injecting anomalies: {e}")

# Create a DataFrame to hold the generated data stream
data_df = pd.DataFrame({"Time": time, "Value": data_stream})

# Validate that the DataFrame was created successfully and contains no missing values
if data_df.isnull().values.any():
    raise ValueError("Data stream contains null values, indicating an issue with data generation.")

# Parameters for the EMA (Exponential Moving Average) with Z-Score anomaly detection
ema_span = 20  # Span or window for the EMA to capture local trend
z_threshold = 2.5  # Z-score threshold to define what constitutes an anomaly

# Validate the parameters for EMA and Z-Score to ensure correct calculations
if ema_span <= 0:
    raise ValueError("ema_span must be a positive integer.")
if z_threshold <= 0:
    raise ValueError("z_threshold must be a positive number.")

# Calculate EMA and rolling standard deviation for Z-score calculation
try:
    data_df['EMA'] = data_df['Value'].ewm(span=ema_span, adjust=False).mean()  # Calculate EMA to smooth data and capture trend
    data_df['RollingStdDev'] = data_df['Value'].rolling(window=ema_span).std()  # Calculate rolling standard deviation
    if data_df['RollingStdDev'].isnull().all():  # Check if rolling std dev contains all NaN values
        raise ValueError("RollingStdDev calculation failed; please check the input data.")
    data_df['RollingStdDev'] = data_df['RollingStdDev'].fillna(data_df['RollingStdDev'].mean())  # Fill NaNs in std dev with mean std dev to avoid errors
except Exception as e:
    raise RuntimeError(f"Error calculating EMA or RollingStdDev: {e}")

# Calculate Z-score for anomaly detection and flag anomalies
try:
    data_df['Z_Score'] = (data_df['Value'] - data_df['EMA']) / data_df['RollingStdDev']  # Calculate Z-score to standardize deviation from EMA
    data_df['Detected_Anomaly'] = data_df['Z_Score'].abs() > z_threshold  # Mark points as anomalies if Z-score exceeds threshold
except ZeroDivisionError:
    raise RuntimeError("RollingStdDev contains zero values, leading to division by zero in Z-score calculation.")
except Exception as e:
    raise RuntimeError(f"Error calculating Z-score or detecting anomalies: {e}")

# Plotting the data stream with EMA and detected anomalies for visualization
try:
    plt.figure(figsize=(14, 6))  # Set figure size for plot
    plt.plot(data_df['Time'], data_df['Value'], label="Data Stream", color="#007FFF")  # Plot main data stream
    plt.plot(data_df['Time'], data_df['EMA'], label="EMA (Trend)", color="#000000", linestyle="--")  # Plot EMA trend line
    plt.scatter(data_df['Time'][data_df['Detected_Anomaly']],
                data_df['Value'][data_df['Detected_Anomaly']],
                color="#E32636", label="Detected Anomalies", marker="x", s=100)  # Highlight anomalies in red
    plt.title("EMA with Z-Score Anomaly Detection")  # Title of the plot
    plt.xlabel("Time")  # Label for x-axis
    plt.ylabel("Value")  # Label for y-axis
    plt.legend()  # Display legend on plot
    plt.show()  # Render plot
except Exception as e:
    raise RuntimeError(f"Error plotting the data: {e}")
