'''
Anomaly Detection Module for Magnetic Field Data
Compares LSTM predictions with actual real-time data to detect anomalies

This module implements a statistical anomaly detection system that:
1. Compares LSTM model predictions with actual magnetic field measurements
2. Calculates the difference (error) between predicted and actual values
3. Uses statistical methods to determine a dynamic threshold
4. Flags data points where the error exceeds the threshold as anomalies
'''

import numpy as np  # For numerical operations (mean, std, etc.)
import pandas as pd  # For data manipulation and time-series operations


class AnomalyDetector:
    """
    Detects anomalies by comparing predicted magnetic field values with actual measurements.
    An anomaly is flagged when the difference exceeds a dynamically calculated threshold.
    
    How it works:
    - The detector learns from historical prediction errors to set an adaptive threshold
    - Uses statistical methods (mean + multiple of standard deviation) to identify outliers
    - Adapts to changing conditions by keeping only recent error history
    """
    
    def __init__(self, threshold_multiplier=2.5, min_samples_for_threshold=10):
        """
        Initialize the anomaly detector with configuration parameters.
        
        Parameters:
        -----------
        threshold_multiplier : float
            Multiplier for standard deviation to set threshold (default: 2.5)
            This means anomalies are detected when error > mean_error + 2.5 * std_error
            Higher values = fewer anomalies detected (more strict)
            Lower values = more anomalies detected (more sensitive)
        min_samples_for_threshold : int
            Minimum number of samples needed to calculate a meaningful threshold
            Before reaching this number, a default threshold is used
        """
        # Store the multiplier used for threshold calculation
        # This determines how many standard deviations away from mean is considered anomalous
        self.threshold_multiplier = threshold_multiplier
        
        # Minimum number of error samples needed before we can calculate statistics
        # Too few samples would give unreliable statistics
        self.min_samples_for_threshold = min_samples_for_threshold
        
        # List to store all historical prediction errors (absolute differences)
        # This grows as we compare more predictions with actual data
        # Used to calculate mean and standard deviation for threshold
        self.prediction_errors = []
        
        # The calculated threshold value (in nanoTesla, nT)
        # Any prediction error above this value is considered an anomaly
        # Initially None, will be calculated after enough samples are collected
        self.anomaly_threshold = None
        
    def calculate_differences(self, actual_times, actual_values, predicted_times, predicted_values):
        """
        Calculate differences between actual and predicted values for overlapping time periods.
        
        This is the core method that:
        1. Interpolates predicted values at exact actual timestamps (more accurate than nearest-neighbor)
        2. Calculates the error (difference) between actual and interpolated predicted values
        3. Updates the error history for threshold calculation
        4. Calculates/updates the anomaly threshold
        5. Marks which points are anomalies
        
        Timestamp Matching Strategy:
        ----------------------------
        Uses linear interpolation to find predicted values at exact actual timestamps.
        This approach is more accurate than nearest-neighbor matching because:
        - Eliminates timing errors (exact time alignment)
        - Uses information from neighboring prediction points
        - Handles different sampling rates gracefully
        - Provides smoother, more accurate comparisons
        
        Only interpolates within the actual prediction time range (no edge tolerance).
        Points outside this range or where interpolation isn't possible are excluded from comparison.
        This prevents "fake" predictions from being created using nearest neighbor fallback.
        
        Parameters:
        -----------
        actual_times : list
            List of datetime objects for actual measurements (from sensors or API)
        actual_values : list
            List of actual magnetic field values (in nanoTesla, nT)
        predicted_times : list
            List of datetime objects for predictions (from LSTM model)
        predicted_values : list
            List of predicted magnetic field values (in nanoTesla, nT)
            
        Returns:
        --------
        differences_df : pandas.DataFrame
            DataFrame with columns: 'time', 'actual', 'predicted', 'difference', 'is_anomaly'
            Each row represents a matched pair of actual and interpolated predicted values
            'predicted' column contains interpolated values at exact actual timestamps
        """
        # Step 1: Check if we have data to compare
        # If either list is empty, return an empty DataFrame
        if not actual_times or not predicted_times:
            return pd.DataFrame(columns=['time', 'actual', 'predicted', 'difference', 'is_anomaly'])
        
        # Step 2: Convert lists to pandas DataFrames for easier manipulation
        # This allows us to use pandas' powerful time-series operations
        actual_df = pd.DataFrame({
            'time': pd.to_datetime(actual_times),  # Ensure times are datetime objects
            'actual': actual_values  # Actual magnetic field measurements
        })
        
        predicted_df = pd.DataFrame({
            'time': pd.to_datetime(predicted_times),  # Ensure times are datetime objects
            'predicted': predicted_values  # Predicted magnetic field values from LSTM
        })
        
        # Step 3: Match actual and predicted data by timestamp using interpolation
        # This approach interpolates predicted values at exact actual timestamps,
        # providing more accurate comparisons than nearest-neighbor matching.
        # 
        # Advantages of interpolation:
        # - Exact time alignment (no timing errors)
        # - Uses information from neighboring prediction points
        # - Handles different sampling rates gracefully
        # - More accurate for anomaly detection
        
        # Sort both dataframes by time (required for interpolation)
        actual_sorted = actual_df.sort_values('time').reset_index(drop=True)
        predicted_sorted = predicted_df.sort_values('time').reset_index(drop=True)
        
        # Set time as index for interpolation
        predicted_indexed = predicted_sorted.set_index('time')
        
        # Define interpolation bounds: only interpolate within the actual prediction time range
        # FIXED: Removed 15-minute tolerance to prevent "fake" predictions at edges
        # Only compare where interpolation is actually possible (between prediction points)
        pred_min_time = predicted_indexed.index.min()
        pred_max_time = predicted_indexed.index.max()
        
        # Filter actual times to only those within the actual prediction range
        # This ensures we only compare where real predictions exist (no edge extrapolation)
        actual_within_range = actual_sorted[
            (actual_sorted['time'] >= pred_min_time) &
            (actual_sorted['time'] <= pred_max_time)
        ].copy()
        
        if len(actual_within_range) == 0:
            # No actual data points within prediction range
            return pd.DataFrame(columns=['time', 'actual', 'predicted', 'difference', 'is_anomaly'])
        
        # Interpolate predicted values at actual timestamps using pandas interpolation
        # Set actual times as index for reindexing
        actual_indexed = actual_within_range.set_index('time')
        
        # Reindex predicted data to actual timestamps and interpolate
        # 'linear' interpolation works well for smoothly varying magnetic fields
        # FIXED: Removed nearest neighbor fallback - only use interpolation
        predicted_interpolated = predicted_indexed.reindex(
            actual_indexed.index
        ).interpolate(method='time')  # Time-based linear interpolation
        
        # FIXED: Removed nearest neighbor fallback that was creating "fake" predictions
        # Now we only use interpolation, which requires at least 2 prediction points
        # Points at the very edges (where interpolation isn't possible) will remain NaN
        # and be filtered out below
        
        # Combine actual and interpolated predicted values
        merged = pd.DataFrame({
            'time': actual_indexed.index,
            'actual': actual_indexed['actual'].values,
            'predicted': predicted_interpolated['predicted'].values
        }).reset_index(drop=True)
        
        # Remove any rows where prediction is still NaN (outside interpolation range or too far)
        # This filters out points where interpolation wasn't possible (too far from prediction data)
        merged = merged.dropna(subset=['predicted'])
        
        # Ensure time column is datetime (should already be, but safety check)
        if len(merged) > 0:
            merged['time'] = pd.to_datetime(merged['time'])
        
        # Step 4: Calculate the error (difference) between actual and predicted values
        # Positive difference means actual > predicted (model underestimated)
        # Negative difference means actual < predicted (model overestimated)
        merged['difference'] = merged['actual'] - merged['predicted']
        
        # We care about the magnitude of error, not the direction
        # So we take the absolute value
        merged['abs_difference'] = abs(merged['difference'])
        
        # Step 5: Update our error history for threshold calculation
        # We keep track of all prediction errors to learn what's "normal"
        if len(merged) > 0:
            # Add all new errors to our history
            self.prediction_errors.extend(merged['abs_difference'].tolist())
            
            # Keep only recent errors (last 1000 points) to adapt to changing conditions
            # This is important because:
            # - Magnetic field behavior might change over time
            # - The LSTM model might improve as it learns
            # - We want the threshold to reflect recent performance, not old performance
            if len(self.prediction_errors) > 1000:
                # Keep only the most recent 1000 errors
                self.prediction_errors = self.prediction_errors[-1000:]
        
        # Step 6: Calculate the anomaly threshold
        # The threshold determines what error value is considered "too large" (anomalous)
        # IMPORTANT: This uses self.threshold_multiplier which can be changed dynamically
        if len(self.prediction_errors) >= self.min_samples_for_threshold:
            # We have enough samples to calculate meaningful statistics
            
            # Calculate the mean (average) of all prediction errors
            # This tells us the typical error size
            mean_error = np.mean(self.prediction_errors)
            
            # Calculate the standard deviation of prediction errors
            # This tells us how much the errors vary from the mean
            # Large std = errors are very variable
            # Small std = errors are consistent
            std_error = np.std(self.prediction_errors)
            
            # Calculate threshold using statistical method:
            # Threshold = Mean Error + (Multiplier × Standard Deviation)
            # This is based on the statistical principle that most data falls within
            # mean ± (multiplier × std). Values beyond this are outliers (anomalies).
            # The multiplier can be changed dynamically by the user
            self.anomaly_threshold = mean_error + self.threshold_multiplier * std_error
            
            # Debug: Log threshold calculation (will be logged by caller)
        else:
            # We don't have enough samples yet, so use a default threshold
            # This is a conservative estimate based on typical magnetic field variations
            # Typical magnetic field variations are in the range of 10-50 nT
            # We use a lower default (10 nT) to be more sensitive when we don't have enough data
            # NOTE: This default doesn't use the multiplier, but once we have enough samples,
            # the threshold will be recalculated using the multiplier
            # If we have some errors but not enough, use a scaled version
            if len(self.prediction_errors) > 0:
                # Use mean of available errors with a small multiplier as a temporary threshold
                temp_mean = np.mean(self.prediction_errors)
                temp_std = np.std(self.prediction_errors) if len(self.prediction_errors) > 1 else temp_mean * 0.1
                self.anomaly_threshold = temp_mean + self.threshold_multiplier * temp_std
            else:
                # No errors yet, use a very low default to catch any differences
                self.anomaly_threshold = 10.0  # Default threshold in nanoTesla (nT)
        
        # Step 7: Mark which data points are anomalies
        # A point is an anomaly if its absolute error exceeds the threshold
        # IMPORTANT: This uses the CURRENT threshold which should reflect the current multiplier
        if self.anomaly_threshold is not None:
            merged['is_anomaly'] = merged['abs_difference'] > self.anomaly_threshold
        else:
            # If threshold is None, mark all as non-anomalies (shouldn't happen)
            merged['is_anomaly'] = False
        
        # Return only the columns we need, excluding 'abs_difference' (internal use only)
        return merged[['time', 'actual', 'predicted', 'difference', 'is_anomaly']]
    
    def detect_anomalies(self, actual_times, actual_values, predicted_times, predicted_values):
        """
        Main method to detect anomalies - this is the primary interface for the detector.
        
        This method:
        1. Calls calculate_differences to compare actual vs predicted data
        2. Filters to keep only the points marked as anomalies
        3. Returns the anomalies along with the threshold used
        
        Parameters:
        -----------
        actual_times : list
            List of datetime objects for actual measurements
        actual_values : list
            List of actual magnetic field values
        predicted_times : list
            List of datetime objects for predictions
        predicted_values : list
            List of predicted magnetic field values
            
        Returns:
        --------
        anomalies_df : pandas.DataFrame
            DataFrame containing ONLY the anomaly points (filtered from all comparisons)
            Columns: 'time', 'actual', 'predicted', 'difference'
            Each row is a detected anomaly
        threshold : float
            The threshold value (in nT) that was used for detection
            Useful for logging and understanding detection sensitivity
        """
        # Step 1: Calculate differences and identify anomalies
        # This does all the heavy lifting: matching, error calculation, threshold calculation
        differences_df = self.calculate_differences(
            actual_times, actual_values, predicted_times, predicted_values
        )
        
        # Step 2: Filter to keep only the anomalies
        # The 'is_anomaly' column is True for anomalies, False for normal points
        # We use boolean indexing to filter: differences_df[differences_df['is_anomaly']]
        # .copy() creates a new DataFrame so we don't modify the original
        anomalies_df = differences_df[differences_df['is_anomaly']].copy()
        
        # Return both the anomalies and the threshold used
        # The threshold is useful for logging and understanding detection sensitivity
        return anomalies_df, self.anomaly_threshold
    
    def get_statistics(self):
        """
        Get statistics about prediction errors and the current threshold.
        
        This is useful for:
        - Monitoring the detector's performance
        - Understanding how the threshold is being calculated
        - Debugging and tuning the detector
        
        Returns:
        --------
        dict : Dictionary with statistics containing:
            - 'mean_error': Average prediction error (in nT)
            - 'std_error': Standard deviation of prediction errors (in nT)
            - 'threshold': Current anomaly threshold (in nT)
            - 'total_samples': Number of error samples used for statistics
        """
        # If we haven't collected any errors yet, return zeros
        if not self.prediction_errors:
            return {
                'mean_error': 0,  # No errors yet, so mean is 0
                'std_error': 0,  # No errors yet, so std is 0
                'threshold': self.anomaly_threshold or 0,  # Use default threshold if set, else 0
                'total_samples': 0  # No samples collected yet
            }
        
        # Calculate and return statistics from collected error history
        return {
            'mean_error': np.mean(self.prediction_errors),  # Average error size
            'std_error': np.std(self.prediction_errors),  # How much errors vary
            'threshold': self.anomaly_threshold or 0,  # Current threshold value
            'total_samples': len(self.prediction_errors)  # How many errors we've seen
        }
