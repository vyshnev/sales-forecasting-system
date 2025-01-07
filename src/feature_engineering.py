import pandas as pd

def create_lagged_features(df, column, lags):
    """Creates lagged features for the specified columns."""

    for lag in lags:
        df[f'{column}_lag_{lag}'] = df.groupby('Store')[column].shift(lag)
    
    return df


def add_time_features(df, date_column):
    """Add time based featres."""
    df[date_column] = pd.to_datetime(df[date_column])
    df['month'] = df[date_column].dt.month
    df['year'] = df[date_column].dt.year
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['weekofyear'] = df[date_column].dt.isocalendar().week.astype(int)
    return df
