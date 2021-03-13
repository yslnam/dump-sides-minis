# Please requirements.txt before using these helper functions
# To import:
# from helper_functions import clean_dataframe, dataframe_summary, eval_missingness, treat_mcar, plot_minmax_boxplot

import math
import numpy as np
import pandas as pd
import datetime as dt
from pandas.api.types import CategoricalDtype
from typing import Tuple, Dict, List
import missingno as msno
import janitor
import pandas_flavor as pf
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import boxplot_stats

def clean_dataframe(df:pd.core.frame.DataFrame, cat_cols:List[str]=None, 
            parse_dates:List[Dict[str, str]]=None, change_types:List[Tuple[str, type]]=None, 
            column_order:List[str]=None, drop_columns:List[str]=None, rename_cols:Dict[str, str]=None,) -> pd.core.frame.DataFrame:
    """
    This function cleans a dataset using pyjanitor functions in one-shot.

    Args:
        df (pd.core.frame.DataFrame): Original dataframe
        cat_cols (List[str], optional): Converts string object rows to categorical to save memory consumption and speed up access. Defaults to None.
        parse_dates (List[Dict[str, str]], optional): Parses and converts data type of a column to datetime format. Defaults to None.
        change_types (List[Tuple[str, type]], optional): Changes data types of a column. Defaults to None.
        column_order (List[str], optional): Reorders dataframe columns by specifying desired order as a list. Defaults to None.
        drop_columns (List[str], optional): Drops columns. Defaults to None.
        rename_cols (Dict[str, str], optional): Renames columns. Defaults to None.

    Returns:
        pd.core.frame.DataFrame: A cleaned dataframe
    """
    cleaned_df = df.copy()

    if cat_cols:
      cleaned_df = cleaned_df.encode_categorical(cat_cols)

    if parse_dates:
      for col_format in parse_dates:
        cleaned_df = cleaned_df.to_datetime(**col_format) # {'column_name':date, 'format':_format}

    if change_types:
      for col_type in change_types:
        cleaned_df = cleaned_df.change_type(*col_type)

    if column_order:
        cleaned_df = cleaned_df.reorder_columns(column_order)

    if drop_columns:
      cleaned_df = cleaned_df.drop(columns=drop_columns)

    if rename_cols:
      cleaned_df = cleaned_df.rename_columns(rename_cols)

    cleaned_df = (
        cleaned_df.clean_names()
        .remove_empty()
    )
    return cleaned_df

def dataframe_summary(df: pd.core.frame.DataFrame, plot_hist:bool=False) -> None:
    """
    This function is my recreation of the skim function in R's skimr package. 
    It takes the name of a dataframe and an option to plot a histogram of numeric features as arguments, 
    and outputs a summary table, along with key descriptive statistics about the dataframe.

    Args:
        df (pd.core.frame.DataFrame): Original dataframe
        plot_hist (bool, optional): If true, plots a histogram of numeric features. Defaults to False.
    """
    # Define numeric and non-numeric (character) columns
    num_col = df.select_dtypes(include=[np.number]).columns
    char_col = df.select_dtypes(exclude=[np.number]).columns

    # Extract descriptive stats
    if list(char_col):
        freq_val = df.describe(exclude=[np.number], datetime_is_numeric=True).transpose()['freq'].values

    if list(num_col):
        mean_val = df.describe(include=[np.number], datetime_is_numeric=True).transpose()['mean'].values
        std_val = df.describe(include=[np.number], datetime_is_numeric=True).transpose()['std'].values

    # Create df summary for character columns
    char_stats = pd.DataFrame.from_records([(col, 
                                             df[col].isnull().sum(),
                                             df[col].notna().sum() / (df[col].isnull().sum() + df[col].notna().sum()),
                                             (df[col] == '').sum(),
                                             df[col].nunique(),
                                             df[col].astype("string").str.isspace().sum()) for col in char_col],
                                           columns=['feature',
                                                    'n_missing',
                                                    'complete_rate',
                                                    'n_empty',
                                                    'n_unique',
                                                    'is_whitespace'])
    char_stats[['freq']] = pd.DataFrame(freq_val)
    char_stats = char_stats.round(2)

    # Create df summary for numeric columns
    num_stats = pd.DataFrame.from_records([(col, 
                                            df[col].isnull().sum(),
                                            df[col].notna().sum() / (df[col].isnull().sum() + df[col].notna().sum())) 
                                           for col in num_col],
                                          columns=['feature',
                                                   'n_missing',
                                                   'complete_rate'])

    if list(num_col):
        num_stats[['mean']] = pd.DataFrame(mean_val)
        num_stats[['std']] = pd.DataFrame(std_val)
    num_stats = num_stats.round(2)

    # Print summary tables
    # Note: Do NOT indent – it will mess up the output formatting!
    print(f'''--- Dataframe Summary ---
Dimensions: {df.shape}\nimport urllib.request
import osl dataframe
    """
    print(msno.matrix(df))
    print(msno.heatmap(df))
    print(msno.dendrogram(df))

def treat_mcar(df:pd.core.frame.DataFrame, col:str, treatment:str) -> pd.core.frame.DataFrame:
    """
    Drops missing values according to pairwise or listwise deletion methods for columns with 
    values that are missing completely at random (MCAR).

    Args:
        df (pd.core.frame.DataFrame): Original dataframe
        col (str): Specified column with MCAR NaNs
        treatment (str): User specifies "pairwise" or "listwise" deletion

    Returns:
        pd.core.frame.DataFrame: Dataframe with treated null values
    """
    if treatment == "pairwise":
        treat_df = df.dropna(subset=[col], how='all', inplace=True)
    elif treatment == "listwise":
        treat_df = df.dropna(subset=[col], how='any', inplace=True)
    return treat_df

def plot_minmax_boxplot(df:pd.core.frame.DataFrame, show_outliers:bool=False, fig_x:int=10, fig_y:int=5) -> None:
    """
    The function plots normalized boxplots for the numeric columns.

    Args:
        df (pd.core.frame.DataFrame): Original dataframe
        show_outliers (bool, optional): If true, shows a list of outliers for each column. Defaults to False.
        fig_x (int, optional): Width ('x') dimensions of a visual. Defaults to 10.
        fig_y (int, optional): Height ('y') dimension so a visual. Defaults to 5.
    """
    # Define numeric features
    num_col = df.select_dtypes(include=[np.number]).columns

    # Apply MinMax normalization and plot histogram
    ax = plt.figure(figsize=(fig_x, fig_y))
    sns.boxplot(x="variable", y="value", 
                data=pd.melt((df[num_col]-df[num_col].min())/(df[num_col].max() - df[num_col].min())))
    plt.title('Data Distribution of Numeric Features')
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout() 
    plt.show();

    # Display an array of outliers
    if show_outliers:
        outliers = boxplot_stats(df[num_col])
        [(col, stats['fliers']) for col, stats in zip(num_col, outliers)]


def correct_round(num):
    """
    This is a function for rounding decimal places of precise latitude
    and longitude coordinates to .75 or .25, per CRU data.
    """
    low = np.floor(num) + 0.25
    high = np.floor(num) + 0.75

    if np.abs(num-low) < np.abs(num-high):
        return low
    return high

@pf.register_dataframe_method
def str_remove(df, column_names: str, pat: str, *args, **kwargs):
    """Remove a substring, given its pattern from a string value, in a given column"""
    for column_name in column_names:
        df[column_name] = df[column_name].str.replace(pat, '', *args, **kwargs)
    return df

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier