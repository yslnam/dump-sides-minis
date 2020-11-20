# Please requirements.txt before using these helper functions
# To import:
# from helper_functions import clean_dataframe, dataframe_summary, eval_missingness, treat_mcar, plot_minmax_boxplot

import numpy as np
import pandas as pd
import datetime as dt
from pandas.api.types import CategoricalDtype
from typing import Tuple, Dict, List
import missingno as msno
import janitor
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import boxplot_stats

def clean_dataframe(df:pd.core.frame.DataFrame, cat_cols:List[str]=None, 
             parse_dates:List[Dict[str, str]]=None, change_types:List[Tuple[str, type]]=None, 
             column_order:List[str]=None, drop_columns:List[str]=None, rename_cols:Dict[str, str]=None,) -> pd.core.frame.DataFrame:
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


def dataframe_summary(df:pd.core.frame.DataFrame, plot_hist:bool=False, 
                      sp_r:int=3, sp_col:int=2, fig_x:int=10, fig_y:int=5) -> None:
    """ 
    This function is a recreation of the skim function in R's
    skimr package. It takes the name of a dataframe and an option
    to plot a histogram of numeric features as arguments, and 
    outputs a summary table, along with key descriptive statistics
    about the dataframe.
    
    plot_hist: If true, plots a histogram of numeric features.
    """ 
    # Define numeric and non-numeric (character) columns
    num_col = df.select_dtypes(include=[np.number]).columns
    char_col = df.select_dtypes(exclude=[np.number]).columns
    
    # Extract descriptive stats
    freq_val = df.describe(exclude=[np.number]).transpose()['freq'].values
    mean_val = df.describe(include=[np.number]).transpose()['mean'].values
    std_val = df.describe(include=[np.number]).transpose()['std'].values
    
    # Create df summary for character columns
    char_stats = pd.DataFrame.from_records([(col, 
                                             df[col].isnull().sum(),
                                             df[col].notna().sum() / (df[col].isnull().sum() + df[col].notna().sum()),
                                             (df[col].values == '').sum(), 
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
    num_stats[['mean']] = pd.DataFrame(mean_val)
    num_stats[['std']] = pd.DataFrame(std_val)
    num_stats = num_stats.round(2)
    
    # Print summary tables
    # Note: Do NOT indent – it will mess up the output formatting!
    print(f'''--- Dataframe Summary ---
Dimensions: {df.shape}\n
--- Data Type Frequency ---
{df.dtypes.value_counts()}\n
--- Summary of Observations by Data Type ---
{"-"*25}
Feature Type: Character
{"-"*25}
{tabulate(char_stats, headers='keys', tablefmt='psql')}\n
{"-"*25}
Feature Type: Numeric
{"-"*25}
{tabulate(num_stats, headers='keys', tablefmt='psql')}''')  
    
    # Print graphs
    if plot_hist:
        fig, axes = plt.subplots(sp_r, sp_col, figsize=(fig_x, fig_y))
        for ax, col in zip(axes.flat, df[num_col]):
            sns.distplot(df[col], ax=ax).set_title(f"Distribution of {col}")
        fig.tight_layout()

def eval_missingness(df:pd.core.frame.DataFrame) -> None:
    """
    Generates all three missingness plots from missingno package.
    """
    print(msno.matrix(df))
    print(msno.heatmap(df))
    print(msno.dendrogram(df))

def treat_mcar(df:pd.core.frame.DataFrame, col:str, treatment:str) -> pd.core.frame.DataFrame:
    """
    Drops missing values according to pairwise or listwise deletion 
    methods for columns with values that are missing completely
    at random (MCAR).
    
    col: specified column with MCAR NaNs
    treatment: user specifies "pairwise" or "listwise" deletion
    """
    if treatment == "pairwise":
        treat_df = df.dropna(subset=[col], how='all', inplace=True)
    elif treatment == "listwise":
        treat_df = df.dropna(subset=[col], how='any', inplace=True)
    return treat_df

def plot_minmax_boxplot(df:pd.core.frame.DataFrame, show_outliers:bool=False, fig_x:int=10, fig_y:int=5) -> None:
    """
    The function plots normalized boxplots for the numeric columns.
    
    show_outliers: If true, shows a list of outliers for each column.
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