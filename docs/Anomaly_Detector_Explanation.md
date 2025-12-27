# Anomaly Detector - Simple Explanation

## What is the Anomaly Detector?

The Anomaly Detector is like a smart quality control inspector for magnetic field measurements. It compares what the AI model **predicted** the magnetic field would be with what was **actually** measured. If the difference is too large, it flags it as an "anomaly" - something unusual that might need attention.

Think of it like this: Imagine you're a teacher grading tests. You have:
- **Predicted scores**: What you expect students to get based on their past performance
- **Actual scores**: What students actually got

If a student's actual score is way different from what you predicted (much higher or lower), that's an anomaly worth investigating!

---

## The Big Picture: How It Works

The anomaly detector follows these simple steps:

1. **Match up the data**: Line up actual measurements with predictions at the same time
2. **Calculate the difference**: See how far off the prediction was
3. **Learn what's normal**: Keep track of typical prediction errors
4. **Set a threshold**: Decide how big an error needs to be to be considered "unusual"
5. **Flag anomalies**: Mark any points where the error exceeds the threshold

---

## Step-by-Step Explanation

### Step 1: Initialization (Setting Up the Detector)

When you create an Anomaly Detector, you can configure two important settings:

- **Threshold Multiplier** (default: 2.5): This controls how sensitive the detector is
  - Higher values (like 3.0) = fewer anomalies detected (more strict, less sensitive)
  - Lower values (like 2.0) = more anomalies detected (more sensitive)
  - Think of it like a security guard: higher multiplier = only really suspicious things get flagged

- **Minimum Samples** (default: 10): How many comparisons are needed before the detector can calculate a meaningful threshold
  - The detector needs to see some data first before it knows what's "normal"
  - Like a new teacher needs to see a few tests before knowing what's a good or bad score

**What gets stored:**
- A list to remember all prediction errors (how wrong the predictions were)
- A threshold value (the line between "normal error" and "anomaly")

---

### Step 2: Matching Timestamps (The Smart Way)

This is where the detector matches actual measurements with predictions. The challenge: timestamps might not match exactly!

**The Old Way (Nearest Neighbor):**
- Find the closest prediction point in time
- Problem: If predictions are at 10:00, 10:05, 10:10 and you have actual data at 10:03, it would just pick 10:00 or 10:05
- This introduces timing errors

**The New Way (Interpolation):**
- For each actual measurement time, calculate what the prediction **would be** at that exact time
- Uses the two nearest prediction points to estimate the value in between
- Like drawing a line between two points and reading the value at any point along that line

**Example:**
- Predictions: 10:00 → 50 nT, 10:05 → 52 nT, 10:10 → 54 nT
- Actual measurement at 10:03
- Interpolation calculates: "At 10:03, the prediction should be about 51.2 nT"
- Now we can compare the actual value at 10:03 with the interpolated prediction at 10:03

**Why this is better:**
- ✅ Exact time alignment (no timing errors)
- ✅ Uses information from multiple prediction points
- ✅ Works even if actual and predicted data have different sampling rates
- ✅ More accurate comparisons

**How Interpolation Works:**
1. **Primary Method**: Uses pandas time-based linear interpolation (`interpolate(method='time')`)
   - Creates a smooth curve through prediction points
   - Estimates values at exact actual timestamps

2. **Fallback for Edge Cases**: If interpolation can't determine a value (at edges), uses nearest neighbor
   - Only if the nearest prediction is within 15 minutes
   - This handles cases where actual data is slightly outside prediction range
   - If too far (>15 minutes), the point is excluded

**Safety checks:**
- Only interpolates within the prediction time range (with 15-minute tolerance at edges)
- If a point is too far from any prediction (>15 minutes), it's excluded (can't reliably interpolate)
- Uses nearest neighbor as fallback only when interpolation fails and point is within tolerance

---

### Step 3: Calculating Differences

Once we have matched pairs (actual value and interpolated prediction at the same time), we calculate:

```
Difference = Actual Value - Predicted Value
```

**What the difference means:**
- **Positive difference**: Actual > Predicted (model underestimated the magnetic field)
- **Negative difference**: Actual < Predicted (model overestimated the magnetic field)
- **Zero difference**: Perfect prediction!

We care about the **magnitude** (size) of the error, not the direction. So we use the absolute value:
```
Absolute Difference = |Actual - Predicted|
```

---

### Step 4: Learning What's Normal

The detector keeps a history of all prediction errors to learn what's "normal" for this system.

**How it works:**
- Every time we compare actual vs predicted, we add the error to our history
- We keep only the most recent 1000 errors (sliding window)
- Why? Because:
  - Magnetic field behavior might change over time
  - The AI model might improve as it learns
  - We want the threshold to reflect recent performance, not old performance

**Example:**
- Error history: [2.1, 1.8, 3.2, 2.5, 1.9, 2.3, ...]
- These represent how "wrong" the predictions typically are
- Most errors are small (1-3 nT), which is normal

---

### Step 5: Calculating the Threshold

The threshold is the line between "normal prediction error" and "anomaly". It's calculated using statistics.

**The Formula:**
```
Threshold = Mean Error + (Multiplier × Standard Deviation)
```

**What this means:**
- **Mean Error**: The average prediction error (typical error size)
- **Standard Deviation**: How much errors vary from the mean
  - Large std = errors are very inconsistent
  - Small std = errors are consistent
- **Multiplier**: How many standard deviations away from mean is considered anomalous (default: 2.5)

**Example:**
- Mean error: 2.0 nT
- Standard deviation: 1.5 nT
- Multiplier: 2.5
- Threshold = 2.0 + (2.5 × 1.5) = 2.0 + 3.75 = **5.75 nT**

This means: "If a prediction error is more than 5.75 nT, it's an anomaly"

**Statistical Principle:**
This is based on the idea that most data falls within a certain range. Values beyond this range are outliers (anomalies). With a multiplier of 2.5, we're saying: "About 99% of normal errors should be below this threshold."

**What if we don't have enough data?**
- If we have fewer than 10 samples, we use a default threshold (10.0 nT)
- Once we have enough samples, we switch to the statistical threshold
- This prevents false alarms early on

---

### Step 6: Marking Anomalies

For each data point, we check:
```
Is Absolute Difference > Threshold?
```

- **Yes** → Mark as anomaly (is_anomaly = True)
- **No** → Mark as normal (is_anomaly = False)

**Example:**
- Actual: 55.0 nT
- Predicted (interpolated): 50.0 nT
- Difference: 5.0 nT
- Threshold: 5.75 nT
- Is 5.0 > 5.75? **No** → Normal (not an anomaly)

- Actual: 58.0 nT
- Predicted (interpolated): 50.0 nT
- Difference: 8.0 nT
- Threshold: 5.75 nT
- Is 8.0 > 5.75? **Yes** → **ANOMALY!**

---

## Main Methods Explained

### `calculate_differences()`

This is the core method that does most of the work:

**Input:**
- Actual measurement times and values
- Predicted times and values

**What it does:**
1. Converts data to pandas DataFrames
2. Interpolates predicted values at exact actual timestamps
3. Calculates differences
4. Updates error history
5. Calculates/updates threshold
6. Marks anomalies

**Output:**
- DataFrame with columns: time, actual, predicted, difference, is_anomaly
- Each row is a comparison point

---

### `detect_anomalies()`

This is the main interface method - the one you call to detect anomalies.

**Input:**
- Same as `calculate_differences()`

**What it does:**
1. Calls `calculate_differences()` to do all the work
2. Filters to keep only the anomalies (where is_anomaly = True)
3. Returns just the anomalies

**Output:**
- DataFrame with only anomaly points
- The threshold value used

---

### `get_statistics()`

This method provides information about the detector's current state.

**Returns:**
- **mean_error**: Average prediction error (in nT)
- **std_error**: Standard deviation of errors (in nT)
- **threshold**: Current anomaly threshold (in nT)
- **total_samples**: Number of error samples collected

**Useful for:**
- Monitoring detector performance
- Understanding how the threshold is calculated
- Debugging and tuning

---

## Real-World Example

Let's walk through a complete example:

**Initialization:**
```python
detector = AnomalyDetector(threshold_multiplier=2.5, min_samples_for_threshold=10)
```

**First batch of data:**
- Actual times: [10:00, 10:01, 10:02, 10:03, 10:04]
- Actual values: [50.0, 50.5, 51.0, 51.5, 52.0] nT
- Predicted times: [10:00, 10:05, 10:10]
- Predicted values: [49.5, 52.5, 55.0] nT

**Step 1: Interpolation**
- At 10:00: Actual=50.0, Predicted=49.5 (exact match)
- At 10:01: Actual=50.5, Predicted=49.8 (interpolated between 49.5 and 52.5)
- At 10:02: Actual=51.0, Predicted=50.1 (interpolated)
- At 10:03: Actual=51.5, Predicted=50.4 (interpolated)
- At 10:04: Actual=52.0, Predicted=50.7 (interpolated)

**Step 2: Calculate differences**
- 10:00: |50.0 - 49.5| = 0.5 nT
- 10:01: |50.5 - 49.8| = 0.7 nT
- 10:02: |51.0 - 50.1| = 0.9 nT
- 10:03: |51.5 - 50.4| = 1.1 nT
- 10:04: |52.0 - 50.7| = 1.3 nT

**Step 3: Update error history**
- Errors: [0.5, 0.7, 0.9, 1.1, 1.3]
- Mean: 0.9 nT
- Std: 0.3 nT
- We have 5 samples, but need 10, so use default threshold: 10.0 nT

**Step 4: Check for anomalies**
- All errors are < 10.0 nT → No anomalies detected

**After more data (now have 15 samples):**
- Mean error: 1.2 nT
- Std error: 0.5 nT
- Threshold: 1.2 + (2.5 × 0.5) = 2.45 nT

**New data point:**
- Actual: 55.0 nT at 10:15
- Predicted (interpolated): 51.0 nT at 10:15
- Difference: |55.0 - 51.0| = 4.0 nT
- Is 4.0 > 2.45? **Yes** → **ANOMALY DETECTED!**

---

## Key Concepts

### Why Interpolation is Better

**Old approach (nearest neighbor):**
- Actual at 10:03 → matches with prediction at 10:00 (3 minutes off)
- This timing error can make normal differences look like anomalies, or hide real anomalies

**New approach (interpolation):**
- Actual at 10:03 → interpolated prediction at exactly 10:03
- Perfect time alignment → more accurate comparisons → better anomaly detection

### Why We Keep Only Recent Errors

Magnetic field behavior and model performance can change over time. By keeping only the last 1000 errors:
- The threshold adapts to current conditions
- Old errors don't skew the threshold
- The detector stays relevant

### Why We Use Statistics

Instead of a fixed threshold (like "anything over 5 nT is an anomaly"), we use statistics because:
- Different conditions have different "normal" error levels
- The threshold adapts automatically
- More reliable than guessing a fixed value

---

## Configuration Tips

### Adjusting Sensitivity

**Want to detect more anomalies?**
- Lower the threshold_multiplier (e.g., 2.0 instead of 2.5)
- More sensitive, but might have more false alarms

**Want fewer false alarms?**
- Raise the threshold_multiplier (e.g., 3.0 instead of 2.5)
- Less sensitive, but might miss some real anomalies

### Minimum Samples

- Default: 10 samples
- Lower values: Threshold calculated sooner, but might be less reliable
- Higher values: More reliable threshold, but takes longer to start detecting

---

## Summary

The Anomaly Detector is a smart system that:

1. **Matches** actual measurements with predictions using interpolation (exact time alignment)
2. **Calculates** how wrong each prediction was
3. **Learns** what's normal by tracking error history
4. **Sets** a dynamic threshold based on statistics
5. **Flags** any points where errors exceed the threshold as anomalies

It's like having a quality control inspector that learns from experience and adapts to changing conditions, helping you identify unusual magnetic field measurements that might indicate problems or interesting phenomena.

---

## Technical Details (For Developers)

### Dependencies
- `numpy`: For numerical operations (mean, std)
- `pandas`: For data manipulation and time-series operations

### Data Flow
1. Input: Lists of times and values (actual and predicted)
2. Processing: Interpolation, difference calculation, threshold calculation
3. Output: DataFrame with matched pairs and anomaly flags

### Performance Considerations
- Interpolation is efficient for time-series data
- Error history limited to 1000 points (O(1) memory)
- Threshold recalculated each time (but simple statistics, very fast)

---

*Document created: 2025*
*Last updated: After interpolation-based matching implementation*

