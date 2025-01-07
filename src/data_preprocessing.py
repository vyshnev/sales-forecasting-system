import pandas as pd

def load_and_merge_data(data_path):
    """Loads, preprocesses and merges all data files."""

    try:
        stores_df = pd.read_csv(f'{data_path}/stores data-set.csv')
        features_df = pd.read_csv(f'{data_path}/Features data set.csv')
        sales_df = pd.read_csv(f'{data_path}/sales data-set.csv')
    except FileNotFoundError:
        print("Error: Could not find one or more data files. Make sure they are in the correct folder and named correctly")
        return None

    # Merge Data
    # convert to datetime before merging
    features_df['Date'] = pd.to_datetime(features_df['Date'],format='mixed')
    sales_df['Date'] = pd.to_datetime(sales_df['Date'], format='mixed')
    # Sum sales across departments and merge
    sales_by_store = sales_df.groupby(['Store', 'Date'])['Weekly_Sales'].sum().reset_index()
    merged_df = pd.merge(sales_by_store, features_df, on=['Store', 'Date'], how='left')
    merged_df = pd.merge(merged_df, stores_df, on=['Store'], how='left')

    return merged_df


def handle_missing_values(df, method='fillna_0'):
  """Handles missing values.
    Args:
        df: Pandas Dataframe
        method: 'fillna_0' or 'mean' or 'median'

  """

  if method == 'fillna_0':
    df = df.fillna(0)
  elif method == 'mean':
    df = df.fillna(df.mean())
  elif method == 'median':
        df = df.fillna(df.median())

  return df


def handle_categorical_data(df, one_hot_encode=True):
  """Handles categorical features such as store type

  Args:
    df: Pandas Dataframe
    one_hot_encode: Bool. If true, the categories will be one hot encoded, otherwise it will be numerical
  """

  if one_hot_encode:
        df = pd.get_dummies(df, columns=['Type'])
  else:
      df['Type'] = df['Type'].astype('category').cat.codes #encode as numeric data

  return df