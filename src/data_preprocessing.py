import pandas as pd

def load_and_merge_data(data_path):
    """ Loads, preprocesses and merge all the data"""

    try:
        store_df = pd.read_csv(f'{data_path}/store data_set.csv')
        feature_df = pd.read_csv(f'{data_path}/Features data set.csv')
        sales_df = pd.read_csv(f'{data_path}/sales data-set.csv')
    except FileNotFoundError:
        print('Error: Could not find one or more files. Make sure they are in the correct path')

        return None

    #Merge the data
    #Convert the date column to datetime
    feature_df['Date'] = pd.to_datetime(feature_df['Date'])
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    # Sum sales across departments and merge
    sales_by_store = sales_df.groupby(['Store', 'Date'])['Weekly_Sales'].sum().reset_index()
    merged_df = pd.merge(sales_by_store, feature_df, on=['Store', 'Date'], how='left')
    merged_df = pd.merge(merged_df, store_df, on=['Store'], how='left')

    return merged_df


def handle_missing_values(df,method='fillna_0'):
    """
    Handling missing values.
    Args:
        df: Panda DataFrame
        method: 'fillna_0' or 'mean' or 'meadian
    """

    if method == 'fillna_0':
        df.fillna(0)
    elif method == 'mean':
        df.fillna(df.mean())
    elif method == 'median':
        df.fillna(df.median())
    
    return df

def handle_categorical_data(df, one_hot_encode=True):
    """
    Handles categorical data such as Stroe type

    Args: 
        df: Panda DataFrame
        one_hot_encode: boolean, If true, the categorical will be one hot encoded, other wise it will be numerical
    """
    if one_hot_encode:
        df = pd.get_dummies(df, columns=['Type'])
    else:
        df['Type'] = df['Type'].astype('category').cat.codes #encodes as numerical data

    return df
