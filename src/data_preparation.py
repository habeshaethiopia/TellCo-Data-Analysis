import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_data(file_path):
    """
    Load the dataset into a Pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(df):
    """
    Clean the dataset by handling missing values and ensuring correct data types.
    """
    # Drop any rows with missing data
    df = df.dropna()
    
    # Ensure the data types are correct (for example, numeric columns should be floats)
    numeric_columns = [
        'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 
        'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
        'Other DL (Bytes)', 'Other UL (Bytes)', 'Total DL (Bytes)', 'Total UL (Bytes)'
    ]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    return df

def check_for_outliers(df):
    """
    Check for outliers in the dataset using boxplots.
    """
    numeric_columns = [
        'Youtube DL (Bytes)', 'Youtube UL (Bytes)', 'Netflix DL (Bytes)', 
        'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 
        'Other DL (Bytes)', 'Other UL (Bytes)', 'Total DL (Bytes)', 'Total UL (Bytes)'
    ]
    
    plt.figure(figsize=(10, 6))
    df[numeric_columns].boxplot()
    plt.title("Boxplot of Data Usage by Category")
    plt.show()

def visualize_correlation(df):
    """
    Visualize the correlation between different data usage columns.
    """
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap of Data Usage Columns")
    plt.show()

def plot_total_data_usage(df):
    """
    Plot total data usage for each entry (Total DL and UL bytes).
    """
    plt.figure(figsize=(10, 6))
    df['Total DL (Bytes)'].plot(kind='bar', color='blue', alpha=0.6, label='Download')
    df['Total UL (Bytes)'].plot(kind='bar', color='red', alpha=0.6, label='Upload')
    plt.title("Total Data Usage (Download vs Upload)")
    plt.xlabel("Index")
    plt.ylabel("Data Usage (Bytes)")
    plt.legend()
    plt.show()

def plot_top_categories(df):
    """
    Visualize top data usage categories (e.g., Gaming, Netflix, Youtube).
    """
    categories = ['Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    total_usage = df[categories].sum()
    
    plt.figure(figsize=(8, 6))
    total_usage.plot(kind='bar', color=['blue', 'green', 'red', 'orange'])
    plt.title("Total Download Data Usage by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Data (Bytes)")
    plt.show()

def plot_interactive(df):
    """
    Create an interactive dashboard using Plotly for total data usage.
    """
    fig = px.scatter(df, x='Total DL (Bytes)', y='Total UL (Bytes)', 
                     color='Last Location Name', 
                     title="Interactive Scatter Plot of Total Data Usage (Download vs Upload)",
                     labels={'Total DL (Bytes)': 'Download (Bytes)', 'Total UL (Bytes)': 'Upload (Bytes)'})
    fig.show()
def get_top_consumers(df, top_n=10):
    """
    Get the top 'n' consumers of data (by total download).
    """
    df['Total Data (Bytes)'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
    top_consumers = df.nlargest(top_n, 'Total Data (Bytes)')
    return top_consumers[['Last Location Name', 'Total Data (Bytes)']]

def get_usage_by_service(df):
    """
    Calculate total data usage by service (e.g., YouTube, Netflix, Gaming).
    """
    services = ['Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    usage_by_service = df[services].sum()
    return usage_by_service

def calculate_growth(df):
    """
    Estimate potential growth by comparing usage between different categories.
    """
    df['Total Download Growth'] = (df['Total DL (Bytes)'] - df['Total UL (Bytes)']) / df['Total UL (Bytes)']
    return df[['Last Location Name', 'Total Download Growth']]

def describe_data(df):
    """
    Get basic statistics about the dataset.
    """
    return df.describe()
