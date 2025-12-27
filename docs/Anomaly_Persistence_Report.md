# Anomaly Persistence Analysis Report

## Question
**If an anomaly has been marked by red vertical line in present cycle, does it remain visible in subsequent cycles as well?**

## Answer: **YES, Anomalies DO Persist Across Cycles** ✅

**UPDATED:** The code has been modified to accumulate anomalies across cycles. Anomalies now remain visible until the application is closed or the limit is reached.

---

## Code Analysis (Updated)

### 1. How Anomalies Are Stored

**Location:** `application.py`, line 907 (initialization)
```python
self.anomaly_times = []
self.anomaly_values = []
```

**Location:** `application.py`, lines 1367-1393 (when anomalies are detected) - **UPDATED**
```python
# FIXED: Accumulate anomalies instead of replacing (retain across cycles)
# Convert existing times to datetime for comparison
existing_times_set = set()
if self.anomaly_times:
    existing_times_set = {pd.to_datetime(t) if not isinstance(t, (datetime, pd.Timestamp)) else t for t in self.anomaly_times}

# Add new anomalies, avoiding duplicates (same timestamp)
new_count = 0
max_anomalies = 1000  # Reasonable limit to prevent memory issues
for new_time, new_value in zip(new_anomaly_times, new_anomaly_values):
    new_time_dt = pd.to_datetime(new_time) if not isinstance(new_time, (datetime, pd.Timestamp)) else new_time
    if new_time_dt not in existing_times_set:
        # If we're at the limit, remove oldest anomaly (FIFO)
        if len(self.anomaly_times) >= max_anomalies:
            self.anomaly_times.pop(0)
            self.anomaly_values.pop(0)
            # Remove from set as well
            if self.anomaly_times:
                oldest_time = pd.to_datetime(self.anomaly_times[0]) if not isinstance(self.anomaly_times[0], (datetime, pd.Timestamp)) else self.anomaly_times[0]
                existing_times_set.discard(oldest_time)
        
        self.anomaly_times.append(new_time_dt)
        self.anomaly_values.append(new_value)
        existing_times_set.add(new_time_dt)
        new_count += 1
```

**Key Observation:** 
- Uses **append**, NOT assignment
- **ACCUMULATES** anomalies across cycles
- Prevents duplicates using a set-based check
- Implements FIFO removal when limit (1000) is reached

---

### 2. When Anomalies Are Cleared - **UPDATED**

**Location:** `application.py`, line 1398-1400 (no valid anomalies found) - **UPDATED**
```python
# FIXED: Don't clear existing anomalies if no new valid anomalies found
# Keep previous anomalies visible
self.log(f'Anomaly Detection: No valid anomalies found in this cycle (all had NaN values after matching). Retaining {len(self.anomaly_times)} previous anomalies.', level='Debug')
```

**Location:** `application.py`, line 1405 (no anomalies detected) - **UPDATED**
```python
# FIXED: Don't clear existing anomalies if no anomalies detected in this cycle
# Keep previous anomalies visible across cycles
stats = self.anomaly_detector.get_statistics()
self.log(f'Anomaly Detection: No anomalies detected in this cycle. Threshold: {threshold:.2f} nT. Mean error: {stats["mean_error"]:.2f} nT. Retaining {len(self.anomaly_times)} previous anomalies.', level='Debug')
```

**Key Observation:**
- **Anomalies are NOT cleared** when no new anomalies are found
- Previous anomalies are **retained** across cycles
- Only cleared when application closes or limit reached

---

### 3. How Anomalies Are Displayed

**Location:** `application.py`, line 1493-1506 (in `_update_canvas()`)
```python
# Remove old vertical lines if they exist
for vline in self.anomaly_vertical_lines:
    try:
        vline.remove()
    except:
        pass
self.anomaly_vertical_lines = []

for vline in self.anomaly_vertical_lines_static:
    try:
        vline.remove()
    except:
        pass
self.anomaly_vertical_lines_static = []
```

**Then creates new lines from ALL accumulated anomalies:**
```python
# Create vertical lines at each anomaly time point
for anomaly_time in anomaly_times_plot:  # Contains ALL accumulated anomalies
    vline_dynamic = self._dynamic_ax.axvline(...)
    self.anomaly_vertical_lines.append(vline_dynamic)
```

**Key Observation:**
- **Old vertical lines are REMOVED first** (to redraw with updated data)
- **New vertical lines are created** from **ALL** accumulated `self.anomaly_times`
- This means **ALL** anomalies (current + historical) are shown

---

### 4. When Detection Happens

**Location:** `application.py`, line 1233
```python
# Perform anomaly detection when we have both actual and predicted data
self._detect_anomalies()
```

**Called from:** `_update_predictions_data()` method
- Triggered when new predictions are available
- Happens periodically as predictions are generated

**What Data Is Used:**
```python
# Line 1262-1263: Only uses NEW real-time data
realtime_actual_times = self.new_x_t if self.new_x_t else []
realtime_actual_values = self.new_y_mag_t if self.new_y_mag_t else []
```

**Key Observation:**
- Only compares **new real-time data** (green line) with predictions
- Does NOT use historical data for detection
- But **accumulates** detected anomalies across cycles

---

## Behavior Summary (Updated)

### Cycle 1: Anomaly Detected
1. `_detect_anomalies()` is called
2. Anomaly found at 10:00 AM
3. `self.anomaly_times.append(10:00 AM)` (ADDS to empty list)
4. Red vertical line drawn at 10:00 AM
5. **Anomaly visible on plot**

### Cycle 2: New Detection (No Anomalies)
1. `_detect_anomalies()` is called again
2. No new anomalies found in this cycle
3. `self.anomaly_times` remains `[10:00 AM]` (NOT cleared)
4. Old vertical lines removed, then redrawn
5. **Previous anomaly (10:00 AM) REMAINS VISIBLE**

### Cycle 3: New Detection (Different Anomaly)
1. `_detect_anomalies()` is called again
2. New anomaly found at 10:15 AM
3. `self.anomaly_times.append(10:15 AM)` (ADDS to existing list)
4. `self.anomaly_times` is now `[10:00 AM, 10:15 AM]`
5. Old vertical lines removed, then redrawn
6. **Both 10:00 AM and 10:15 AM anomalies visible**

### Cycle 4: Duplicate Prevention
1. `_detect_anomalies()` is called again
2. Same anomaly at 10:00 AM detected again (duplicate)
3. System checks: `10:00 AM in existing_times_set?` → YES
4. **Duplicate skipped** (not added again)
5. `self.anomaly_times` remains `[10:00 AM, 10:15 AM]`
6. **No duplicate lines drawn**

### After 1000 Anomalies: FIFO Removal
1. `_detect_anomalies()` is called
2. New anomaly found
3. `len(self.anomaly_times) >= 1000` → TRUE
4. Oldest anomaly removed: `self.anomaly_times.pop(0)`
5. New anomaly added: `self.anomaly_times.append(new_time)`
6. **Total remains at 1000, oldest removed, newest added**

---

## Code Flow Diagram (Updated)

```
Cycle 1:
_detect_anomalies() called
  ↓
Anomaly found at 10:00 AM
  ↓
self.anomaly_times.append(10:00 AM)  ← ADDS to []
  ↓
self.anomaly_times = [10:00 AM]
  ↓
_update_canvas() called
  ↓
Remove old lines (none)
  ↓
Draw red line at 10:00 AM
  ↓
✅ Anomaly visible

Cycle 2 (20 seconds later):
_detect_anomalies() called again
  ↓
No anomalies found
  ↓
self.anomaly_times = [10:00 AM]  ← RETAINED (not cleared)
  ↓
_update_canvas() called
  ↓
Remove old lines (removes 10:00 AM line)
  ↓
Redraw ALL anomalies from self.anomaly_times
  ↓
Draw red line at 10:00 AM again
  ↓
✅ 10:00 AM anomaly STILL VISIBLE

Cycle 3:
_detect_anomalies() called again
  ↓
New anomaly found at 10:15 AM
  ↓
self.anomaly_times.append(10:15 AM)  ← ADDS to [10:00 AM]
  ↓
self.anomaly_times = [10:00 AM, 10:15 AM]
  ↓
_update_canvas() called
  ↓
Remove old lines
  ↓
Redraw ALL anomalies
  ↓
Draw red lines at 10:00 AM and 10:15 AM
  ↓
✅ Both anomalies visible
```

---

## Features of the Updated Implementation

### 1. **Accumulation**
- Anomalies are **added** to the list, not replaced
- Each cycle adds new anomalies to existing ones
- Historical anomalies are preserved

### 2. **Duplicate Prevention**
- Uses a set (`existing_times_set`) to track existing timestamps
- Checks if timestamp already exists before adding
- Prevents the same anomaly from appearing multiple times

### 3. **Memory Management**
- Maximum limit: 1000 anomalies
- FIFO (First In, First Out) removal when limit reached
- Oldest anomalies removed first
- Prevents memory issues during long-running sessions

### 4. **Persistence**
- Anomalies remain visible across all cycles
- Only cleared when:
  - Application closes
  - Limit reached (oldest removed, but others remain)
- No clearing on empty detection cycles

### 5. **Visualization**
- All accumulated anomalies are redrawn each cycle
- Ensures visibility even if plot is zoomed/panned
- Maintains consistency across plot updates

---

## Benefits of the Updated Design

### Advantages:
- ✅ **Historical Tracking:** See anomaly trends over time
- ✅ **Better Analysis:** Compare anomalies across different cycles
- ✅ **No Data Loss:** Important anomalies remain visible
- ✅ **Memory Safe:** Automatic limit prevents issues
- ✅ **Duplicate Prevention:** Same anomaly won't appear multiple times
- ✅ **Comprehensive View:** See all anomalies in one view

### Trade-offs:
- ⚠️ Plot can become cluttered with many anomalies (but limit prevents excessive clutter)
- ⚠️ Memory usage grows (but capped at 1000 anomalies)
- ✅ Overall: Benefits outweigh trade-offs

---

## Configuration

**Memory Limit (Line 1375):**
```python
max_anomalies = 1000  # Reasonable limit to prevent memory issues
```

**To Adjust:**
- Increase for longer history (more memory)
- Decrease for less memory usage
- Current value (1000) is a good balance

---

## Conclusion

**Answer: YES, anomalies DO persist across cycles.**

**Evidence:**
1. Line 1388 uses `append()` not assignment - **accumulates** anomalies
2. Lines 1398-1400 and 1405 do NOT clear anomalies - **retains** previous anomalies
3. Duplicate prevention ensures no duplicates
4. Memory limit prevents excessive accumulation
5. All accumulated anomalies are redrawn each cycle

**Current Behavior:**
- Anomalies are **persistent** - remain visible across all cycles
- New anomalies are **added** to existing ones
- Duplicates are **prevented**
- Memory is **managed** (1000 limit with FIFO)
- Historical anomalies are **preserved** for analysis

**This allows:**
- Long-term anomaly tracking
- Trend analysis
- Comprehensive anomaly visualization
- Better understanding of anomaly patterns over time

---

*Report generated: 2025*
*Code analyzed: application.py, lines 907, 1233, 1367-1393, 1398-1400, 1405, 1493-1534*
*Last updated: After anomaly persistence fix implementation*
