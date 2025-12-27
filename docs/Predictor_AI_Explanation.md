# LSTM Predictor - Simple Explanation

## What is the LSTM Predictor?

The LSTM Predictor is like a crystal ball for magnetic fields. It uses Artificial Intelligence (specifically a type of neural network called LSTM - Long Short-Term Memory) to predict what the magnetic field will be in the future based on what it has been in the past.

Think of it like weather forecasting: just as meteorologists use past weather patterns to predict tomorrow's weather, the LSTM Predictor uses past magnetic field measurements to predict future values.

---

## The Big Picture: How It Works

The predictor follows these steps:

1. **Learn from history**: Train the AI model on past magnetic field data
2. **Look at recent patterns**: Use the last few data points as context
3. **Predict the future**: Generate predictions for upcoming time points
4. **Keep learning**: Continuously update the model as new data arrives (optional)

---

## Step-by-Step Explanation

### Step 1: Initialization (Setting Up the Predictor)

When you create an LSTM Predictor, you configure several important settings:

**Parameters:**

- **window_size** (default: 5): How many past data points the model looks at to make a prediction
  - Like looking at the last 5 days of weather to predict tomorrow
  - Larger window = more context, but slower processing
  - Smaller window = faster, but might miss long-term patterns

- **initial_train_points** (default: 3400): How many historical data points to use for initial training
  - The model needs to learn from examples before it can predict
  - More training data = better predictions (usually)
  - But too much = slower training

- **epochs_per_update** (default: 5): How many times the model reviews the training data
  - Each "epoch" is one complete pass through the training data
  - More epochs = better learning, but slower
  - Think of it like studying: more review sessions = better understanding

- **learning_rate** (default: 0.001): How fast the model learns
  - Too high = learns too fast, might miss important details
  - Too low = learns too slow, takes forever
  - This is a "Goldilocks" parameter - needs to be just right

- **update_training** (default: True): Whether to keep learning as new data arrives
  - True = model improves over time (adaptive learning)
  - False = model stays fixed after initial training

**What gets created:**
- A scaler (MinMaxScaler) to normalize data (convert to 0-1 range)
- An empty model (will be built when needed)

---

### Step 2: Creating Windowed Dataset

This is like preparing flashcards for studying. The model needs data in a specific format.

**The Problem:**
- We have a long list of magnetic field values over time
- The model needs pairs: (past values, future value)

**The Solution:**
For a window size of 5:
- Input: [value1, value2, value3, value4, value5]
- Output: value6

Then slide the window:
- Input: [value2, value3, value4, value5, value6]
- Output: value7

And so on...

**Example:**
If we have data: [10, 12, 14, 16, 18, 20, 22, ...]

With window_size=3:
- Training pair 1: Input=[10,12,14], Output=16
- Training pair 2: Input=[12,14,16], Output=18
- Training pair 3: Input=[14,16,18], Output=20
- etc.

This creates many training examples from one long sequence!

---

### Step 3: Building the Model

The model architecture is like a brain with specific layers:

**Layer 1: LSTM (32 units)**
- This is the "memory" layer
- Remembers patterns from the input sequence
- 32 units = 32 different patterns it can learn
- `return_sequences=False` means it only outputs the final prediction

**Layer 2: Dense (16 units, ReLU activation)**
- This is a "thinking" layer
- Processes the LSTM output
- ReLU activation = only positive values pass through (like a filter)

**Layer 3: Dense (1 unit)**
- This is the "output" layer
- Produces the final prediction (single value)

**Optimizer: Adam**
- This is the "learning algorithm"
- Adam is good at finding the best model parameters
- Uses the learning_rate to control how fast it learns

**Loss Function: Mean Squared Error**
- This measures how wrong the predictions are
- The model tries to minimize this during training
- Lower error = better predictions

---

### Step 4: Data Normalization (Scaling)

Magnetic field values can be large numbers (like 50,000 nT). Neural networks work better with smaller numbers (0-1 range).

**Why normalize?**
- Prevents large numbers from dominating
- Makes training faster and more stable
- Like converting temperatures from Fahrenheit to a 0-1 scale

**How it works:**
- Find the minimum and maximum values in the data
- Scale everything to 0-1 range:
  ```
  normalized_value = (value - min) / (max - min)
  ```
- When predicting, convert back:
  ```
  actual_value = normalized_value * (max - min) + min
  ```

**Example:**
- Original data: [48000, 50000, 52000]
- Min = 48000, Max = 52000
- Normalized: [0.0, 0.5, 1.0]
- After prediction, convert back to original scale

---

### Step 5: Initial Training

Before making predictions, the model needs to learn from historical data.

**Process:**
1. Take the first `initial_train_points` data points
2. Create windowed dataset from this data
3. Train the model for `epochs_per_update` epochs
4. Model now "knows" the patterns in the data

**What happens during training:**
- Model makes predictions on training data
- Compares predictions with actual values
- Calculates error (loss)
- Adjusts internal parameters to reduce error
- Repeats for specified number of epochs

**Example:**
- Initial data: 5000 points
- initial_train_points: 3400
- Use first 3400 points for training
- Create ~3395 training pairs (window_size=5)
- Train for 5 epochs
- Model is now ready to predict!

---

### Step 6: Making Predictions (Forecasting)

This is where the magic happens! The model predicts future values.

**The Process:**

1. **Start with recent data:**
   - Take the last `window_size` points from training data
   - This is the "context" for the first prediction

2. **Predict one step ahead:**
   - Feed the window to the model
   - Model outputs a prediction
   - Convert from normalized scale back to original scale

3. **Update the window:**
   - Remove the oldest value
   - Add the new prediction
   - Window slides forward in time

4. **Repeat for n_future steps:**
   - Each prediction becomes input for the next
   - This is called "autoregressive" prediction
   - Like a chain: each link depends on the previous one

5. **Optional: Update training:**
   - If `update_training=True`, add the prediction to training data
   - Retrain the model with this new data
   - Model adapts to new patterns

**Example Walkthrough:**

Starting data: [10, 12, 14, 16, 18] (last 5 points)
Window size: 5
Predict 3 future values

**Step 1:**
- Input window: [10, 12, 14, 16, 18]
- Model predicts: 20
- New window: [12, 14, 16, 18, 20]

**Step 2:**
- Input window: [12, 14, 16, 18, 20]
- Model predicts: 22
- New window: [14, 16, 18, 20, 22]

**Step 3:**
- Input window: [14, 16, 18, 20, 22]
- Model predicts: 24
- Predictions: [20, 22, 24]

---

### Step 7: Generating Future Timestamps

Predictions need timestamps! The model calculates when each prediction should occur.

**Process:**
1. Look at the last two timestamps in input data
2. Calculate the time difference (delta)
3. For each future prediction, add (i+1) × delta to the last timestamp

**Example:**
- Last timestamp: 2025-01-15 10:00:00
- Second-to-last: 2025-01-15 09:59:00
- Delta: 1 minute
- Predictions:
  - Prediction 1: 2025-01-15 10:01:00 (last + 1×delta)
  - Prediction 2: 2025-01-15 10:02:00 (last + 2×delta)
  - Prediction 3: 2025-01-15 10:03:00 (last + 3×delta)

**Edge case:**
- If only one timestamp, assume 1 second intervals

---

## Key Concepts Explained

### What is LSTM?

**LSTM (Long Short-Term Memory)** is a special type of neural network designed to remember patterns over time.

**Why LSTM for time series?**
- Regular neural networks forget previous inputs
- LSTM has "memory cells" that remember important information
- Perfect for sequences like time series data

**Analogy:**
- Regular network: Like someone who only remembers the last thing you said
- LSTM: Like someone who remembers the conversation context

### Autoregressive Prediction

**What it means:**
- Each prediction uses previous predictions as input
- Like a chain reaction: one prediction leads to the next

**Why it's powerful:**
- Can predict far into the future
- Maintains temporal relationships
- Captures long-term dependencies

**Limitation:**
- Errors can accumulate over time
- Predictions get less accurate the further ahead you go

### Adaptive Learning (update_training=True)

**What it does:**
- Model keeps learning as new data arrives
- Adapts to changing patterns
- Improves over time

**How it works:**
1. Make a prediction
2. Add prediction to training data
3. Retrain model with updated data
4. Model now "knows" about this new pattern

**Benefits:**
- Handles non-stationary data (patterns that change over time)
- Improves accuracy as more data arrives
- Adapts to new conditions

**Trade-off:**
- Slower (retraining takes time)
- More computationally expensive

---

## Real-World Example

Let's walk through a complete example:

**Setup:**
```python
predictor = LSTMPredictor(
    window_size=15,
    initial_train_points=5000,
    epochs_per_update=10,
    learning_rate=0.001,
    update_training=True
)
```

**Input Data:**
- 5000 historical magnetic field measurements
- Timestamps: Every 1 minute
- Values: Range from 48,000 to 52,000 nT

**Step 1: Normalization**
- Min: 48,000 nT
- Max: 52,000 nT
- All values scaled to 0-1 range

**Step 2: Initial Training**
- Use first 5000 points
- Create ~4985 training pairs (window_size=15)
- Train for 10 epochs
- Model learns patterns in the data

**Step 3: Prediction**
- Last 15 points: [49,500, 49,520, 49,540, ..., 49,700]
- Predict 100 future values
- Each prediction uses previous predictions

**Step 4: Timestamps**
- Last timestamp: 2025-01-15 10:00:00
- Delta: 1 minute
- Future timestamps: 10:01, 10:02, 10:03, ..., 11:39

**Step 5: Output**
- 100 predicted values
- 100 corresponding timestamps
- Saved to CSV file

---

## How It's Used in the Application

**In application.py:**

1. **Data Collection:**
   - Application collects real-time magnetic field data
   - Saves to `predict_input.csv` in session folder

2. **Process Launch:**
   - Application starts `predictor_ai.py` as a separate process
   - Passes input file path as argument
   - Runs in background (doesn't freeze GUI)

3. **Prediction Generation:**
   - `predictor_ai.py` reads input file
   - Creates predictor instance
   - Generates 100 future predictions
   - Saves to `predict_out.csv`

4. **Result Reading:**
   - Application periodically checks for output file
   - Reads predictions when ready
   - Displays on plot (purple line)
   - Uses for anomaly detection

**Why separate process?**
- Prediction can take time (especially with training)
- Running in separate process keeps GUI responsive
- Can run on different CPU core (parallel processing)

---

## Configuration Tips

### Window Size

**Small window (5-10):**
- ✅ Faster training and prediction
- ✅ Good for short-term patterns
- ❌ Might miss long-term trends

**Large window (20-30):**
- ✅ Captures long-term patterns
- ✅ Better for complex sequences
- ❌ Slower, more memory needed

**Recommendation:** Start with 15, adjust based on data characteristics

### Initial Training Points

**Too few (< 1000):**
- ❌ Model doesn't learn well
- ❌ Poor predictions

**Too many (> 10000):**
- ❌ Very slow training
- ❌ Diminishing returns

**Recommendation:** 3000-5000 points is usually good

### Learning Rate

**Too high (> 0.01):**
- ❌ Model might overshoot optimal values
- ❌ Unstable training

**Too low (< 0.0001):**
- ❌ Very slow learning
- ❌ Might get stuck

**Recommendation:** 0.001 is a good starting point

### Epochs Per Update

**Few epochs (1-3):**
- ✅ Fast training
- ❌ Might not learn enough

**Many epochs (10-20):**
- ✅ Better learning
- ❌ Slower, might overfit

**Recommendation:** 5-10 epochs is usually sufficient

---

## Common Issues and Solutions

### Issue: Predictions are way off

**Possible causes:**
- Not enough training data
- Window size too small/large
- Learning rate wrong
- Data has sudden changes (model can't predict)

**Solutions:**
- Increase initial_train_points
- Try different window sizes
- Adjust learning rate
- Check if data is predictable (some patterns are inherently unpredictable)

### Issue: Predictions are too similar (no variation)

**Possible causes:**
- Model is too conservative
- Not learning patterns well
- Overfitting to training data

**Solutions:**
- Increase epochs_per_update
- Try different learning rate
- Check if training data has enough variation

### Issue: Process takes too long

**Possible causes:**
- Too many training points
- Too many epochs
- update_training=True (retrains every step)

**Solutions:**
- Reduce initial_train_points
- Reduce epochs_per_update
- Set update_training=False

---

## Summary

The LSTM Predictor is a powerful AI tool that:

1. **Learns** from historical magnetic field data
2. **Remembers** patterns using LSTM neural networks
3. **Predicts** future values using autoregressive forecasting
4. **Adapts** to new data (if enabled)
5. **Outputs** predictions with timestamps for visualization and anomaly detection

It's like having a smart assistant that watches magnetic field patterns and tells you what's likely to happen next!

---

## Technical Details (For Developers)

### Dependencies
- `tensorflow`: For LSTM neural network
- `numpy`: For numerical operations
- `pandas`: For data handling
- `sklearn`: For data normalization (MinMaxScaler)

### Model Architecture
- Input shape: (window_size, 1)
- LSTM layer: 32 units
- Dense layer: 16 units (ReLU)
- Output layer: 1 unit (linear)
- Total parameters: ~2,000-3,000 (depends on window_size)

### Performance
- Training time: Depends on data size and epochs (seconds to minutes)
- Prediction time: Very fast (~milliseconds per prediction)
- Memory usage: Moderate (model + data in memory)

### File I/O
- Input: CSV file with 'x' (timestamps) and 'y' (values) columns
- Output: CSV file with 'x' (future timestamps) and 'y' (predictions) columns
- Both files in same directory (session folder)

---

*Document created: 2025*
*Last updated: After thorough code review*

