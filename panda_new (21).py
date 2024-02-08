#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    return df

# Function to generate summary statistics including Quantile statistics
def generate_summary_statistics(df):
    st.subheader("Dataset Statistics")
    dataset_stats = df.describe().transpose()
    st.write(dataset_stats)

    st.subheader("Quantile Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary_stats = df[col].describe(percentiles=[.05, .25, .5, .75, .95])
        iqr = summary_stats['75%'] - summary_stats['25%']
        quantile_stats = pd.DataFrame({
            'Minimum': [summary_stats['min']],
            '5th Percentile': [summary_stats['5%']],
            'Q1 (25th Percentile)': [summary_stats['25%']],
            'Median (50th Percentile)': [summary_stats['50%']],
            'Q3 (75th Percentile)': [summary_stats['75%']],
            '95th Percentile': [summary_stats['95%']],
            'Maximum': [summary_stats['max']],
            'Range': [summary_stats['max'] - summary_stats['min']],
            'Interquartile Range (IQR)': [iqr]
        }, index=[col])
        st.write(quantile_stats)

    st.subheader("Descriptive Statistics")
    descriptive_stats = pd.DataFrame({
        'Mean': df.mean(),
        'Standard Deviation': df.std(),
        'Coefficient of Variation (CV)': df.std() / df.mean(),
        'Kurtosis': df.kurtosis(),
        'Median Absolute Deviation (MAD)': df.mad(),
        'Skewness': df.skew(),
        'Sum': df.sum(),
        'Variance': df.var(),
        'Monotonicity': df.apply(lambda col: col.is_monotonic).rename('Is Monotonic')
    })
    st.write(descriptive_stats)

# Function to display missing values
def display_missing_values(df):
    st.subheader("Missing Values")
    missing_data = df.isnull().sum()
    missing_data_percentage = (missing_data / len(df)) * 100
    missing_info = pd.concat([missing_data, missing_data_percentage], axis=1, keys=['Total', 'Percentage'])
    st.write(missing_info)

# Function to generate correlation matrix heatmap
def generate_correlation_heatmap(df):
    st.subheader("Correlation Matrix and Heatmap")

    # Select variables for the heatmap using multiselect
    selected_variables = st.multiselect('Select variables for heatmap', df.columns)

    # Generate correlation matrix heatmap for selected variables
    if len(selected_variables) > 0:
        selected_df = df[selected_variables]
        corr_matrix = selected_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
        plt.title('Correlation Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(plt.gcf())
    else:
        st.write("Please select at least one variable for the heatmap.")

# Function to generate histograms with annotations
def generate_histogram(df, col):
    st.subheader(f"Histogram for {col}")
    plt.figure(figsize=(8, 6))
    ax = sns.histplot(df[col], kde=True)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 5),
                    textcoords='offset points')
        ax.annotate(f'{p.get_x():.2f}-{p.get_x() + p.get_width():.2f}', (p.get_x() + p.get_width() / 2., 0),
                    ha='center', va='bottom', fontsize=7, color='black', xytext=(0, 5),
                    textcoords='offset points', rotation=20)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    st.pyplot(plt.gcf())

# Function to generate scatter plot
def generate_scatter_plot(df):
    st.subheader("Scatter Plot Interaction")

    # Dropdowns for selecting variables for scatter plots
    x_variable = st.selectbox('Select variable for X-axis', df.columns)
    y_variable = st.selectbox('Select variable for Y-axis', df.columns)

    # Generate scatter plot
    st.subheader(f"Scatter Plot for {x_variable} vs {y_variable}")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_variable, y=y_variable, data=df)
    st.pyplot(plt.gcf())

# Main function
def main():
    st.title("Streamlit YData Profiling")

    # File upload
    uploaded_file = st.file_uploader("Upload XLSX file", type=["xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        # Generate summary statistics
        generate_summary_statistics(df)

        # Display missing values
        display_missing_values(df)

        # Generate correlation matrix heatmap
        generate_correlation_heatmap(df)

        # Allow users to choose a particular histogram to draw
        histogram_variable = st.selectbox('Select variable for histogram', df.columns)

        # Generate histogram
        generate_histogram(df, histogram_variable)

        # Generate scatter plot
        generate_scatter_plot(df)

# Run the app
if __name__ == "__main__":
    main()









# In[ ]:




