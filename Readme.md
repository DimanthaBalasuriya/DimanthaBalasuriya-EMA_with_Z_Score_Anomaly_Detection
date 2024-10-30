
# EMA with Z-Score Anomaly Detection

This section provides a clear overview of the algorithm and the approach used. It outlines the key steps involved, the rationale behind the chosen methods, and highlights specific considerations taken into account during the development process.

The goal of this algorithm is to detect unusual patterns, known as anomalies, in a stream of data that typically follows a regular trend. Think of it as noticing sudden spikes or dips in what is usually a steady line of information—like recognizing if someone begins to speed up and slow down randomly on a normally smooth road.

To start, I develop a data pattern that includes a repeating cycle (similar to seasons), a slow and steady increase (the trend), along with some random variations (noise). This combination simulates real-life data features, such as annual temperatures or website visits that grow gradually, with seasonal peaks.

To represent unexpected events, I introduce random "spikes" at certain points. These spikes resemble sudden, unusual occurrences, like a significant jump in sales during a promotional event.

To understand the normal trend, the algorithm calculates an "Exponential Moving Average" (EMA), which adapts to the general trend of the data as new inputs are received. The EMA functions like a flexible line that tracks the overall direction of the data without being overly influenced by each minor fluctuation.

Next, the algorithm calculates a measure of how much each point deviates from the EMA, using a statistic known as the "Z-score." A Z-score indicates how far a data point is from the average trend line. If a point has a high Z-score, it signifies that it is unusually distant from what the trend predicts.

If a point's Z-score exceeds a set threshold (for example, 2.5), it is flagged as an anomaly. This threshold indicates that the point is sufficiently far from the trend to warrant further investigation, akin to noticing an increase in your heartbeat during a workout.

The algorithm outputs a list of points identified as anomalies—those that do not fit the typical pattern. These flagged points may indicate noteworthy events, such as errors or significant changes in the underlying data process.

By combining the trend (EMA) with variability (Z-score), this algorithm identifies anomalies based on how far data points stray from a baseline pattern, even as the overall pattern gradually changes. It efficiently detects unusual data in settings where new information continuously streams in, such as financial transactions or sensor readings. This method ensures that only truly unusual values are flagged, avoiding distraction from regular fluctuations or slow changes.

**Issues with the Current Solution**

- Recalculating for each new point in large datasets can be slow. Consider using more efficient libraries or an incremental variance method.
- Repeated EMA calculations may introduce overhead. Using simpler math libraries or hardware acceleration could help mitigate this.
- Frequent probability calculations may slow processing in large datasets. Batch injecting anomalies could optimize performance.
- Chained calculations in pandas can add latency. Perform calculations outside the DataFrame and then add the results afterward.
- Static plotting is not suitable for real-time data streams. Utilize interactive libraries like Plotly or a lightweight dashboard instead.
- Storing large, continuous datasets in the DataFrame consumes excessive memory. Implement a sliding window to limit memory usage to recent points only.


**There are various algorithms available to diagnose anomalies in data streams, including**

- Incremental or Online Algorithms
- Statistical Process Control (SPC)
- Machine Learning Models
- Approximate Nearest Neighbor (ANN) Search
- Fourier or Wavelet Transforms
- Sliding Window with Summaries

Most of these solutions typically require more memory and CPU power than the currently implemented solution. However, they can be more accurate in different scenarios.
