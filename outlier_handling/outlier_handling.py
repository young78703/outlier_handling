from scipy.stats import rankdata

def drop_outliers_by_percentiles(data, column, lower_percentile, upper_percentile):
    """
    Drops rows from a Pandas DataFrame based on percentiles of a given column.
    
    Parameters:
    data (pandas.DataFrame): The input data.
    column (str): The name of the column to use for computing percentiles.
    lower_percentile (float): The lower percentile bound (between 0 and 100).
    upper_percentile (float): The upper percentile bound (between 0 and 100).
    
    Returns:
    pandas.DataFrame: The modified DataFrame with outliers dropped.
    """
    # Check input arguments
    if column not in data.columns:
        raise ValueError("Column '%s' not found in data." % column)
    if not (0 <= lower_percentile <= 100):
        raise ValueError("Lower percentile bound must be between 0 and 100.")
    if not (0 <= upper_percentile <= 100):
        raise ValueError("Upper percentile bound must be between 0 and 100.")
    
    # Compute percentiles
    percentiles = pd.Series((rankdata(data[column]) / len(data)) * 100)
    
    # Drop outliers outside bounds
    mask = (percentiles >= upper_percentile) | (percentiles <= lower_percentile)
    return data.loc[~mask]

from scipy import stats

def drop_outliers_by_zscores(data, column, lower_zscore, upper_zscore):
    """
    Drops rows from a Pandas DataFrame based on z-scores of a given column.
    
    Parameters:
    data (pandas.DataFrame): The input data.
    column (str): The name of the column to use for computing z-scores.
    lower_zscore (float): The lower z-score boundary.
    upper_zscore (float): The upper z-score boundary.
    
    Returns:
    pandas.DataFrame: The modified DataFrame with outliers dropped.
    """
    # Check input arguments
    if column not in data.columns:
        raise ValueError("Column '%s' not found in data." % column)
    if not np.isfinite(lower_zscore):
        raise ValueError("Lower z-score boundary must be finite.")
    if not np.isfinite(upper_zscore):
        raise ValueError("Upper z-score boundary must be finite.")
    
    # Compute z-scores
    z_scores = pd.Series(stats.zscore(data[column]), index=data.index)
    
    # Drop outliers outside boundaries
    mask = (z_scores >= upper_zscore) | (z_scores <= lower_zscore)
    return data.loc[~mask]

def clip_outliers_by_zscores(data, column, upper_zscore, lower_zscore):
    """
    Clips the outliers of a column in a Pandas DataFrame based on z-scores.
    
    Parameters:
    data (pandas.DataFrame): The input data.
    column (str): The name of the column to clip.
    lower_zscore (float): The lower z-score boundary.
    upper_zscore (float): The upper z-score boundary.
    
    Returns:
    pandas.DataFrame: The modified DataFrame with outliers clipped.
    """
    # Check input arguments
    if column not in data.columns:
        raise ValueError("Column '%s' not found in data." % column)
    if not np.isfinite(lower_zscore):
        raise ValueError("Lower z-score boundary must be finite.")
    if not np.isfinite(upper_zscore):
        raise ValueError("Upper z-score boundary must be finite.")
    
    # Compute mean and standard deviation
    mean = np.mean(data[column])
    std_dev = np.std(data[column])
    
    # Compute lower and upper value bounds based on z-scores
    lower_value = lower_zscore * std_dev + mean
    upper_value = upper_zscore * std_dev + mean
    
    # Clip outliers
    data_clipped = data.copy()
    data_clipped[column] = data_clipped[column].clip(lower_value, upper_value)
    
    return data_clipped

def clip_outliers_by_percentiles(data, column, lower_percentile, upper_percentile):
    """
    Clips the outliers of a column in a Pandas DataFrame based on percentiles.
    
    Parameters:
    data (pandas.DataFrame): The input data.
    column (str): The name of the column to clip.
    lower_percentile (float): The lower percentile bound (between 0 and 100).
    upper_percentile (float): The upper percentile bound (between 0 and 100).
    
    Returns:
    pandas.DataFrame: The modified DataFrame with outliers clipped.
    """
    # Check input arguments
    if column not in data.columns:
        raise ValueError("Column '%s' not found in data." % column)
    if not (0 <= lower_percentile <= 100):
        raise ValueError("Lower percentile bound must be between 0 and 100.")
    if not (0 <= upper_percentile <= 100):
        raise ValueError("Upper percentile bound must be between 0 and 100.")
    
    # Compute percentiles
    p_upper = np.percentile(data[column], upper_percentile)
    p_lower = np.percentile(data[column], lower_percentile)
    
    # Clip outliers
    data[column] = data[column].clip(p_lower, p_upper)
    return data