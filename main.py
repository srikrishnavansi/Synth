import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from io import BytesIO

# Function to generate synthetic data for selected columns
def generate_synthetic_data(real_data, selected_columns):
    # Identify continuous and categorical features
    continuous_features = []
    categorical_features = []

    for column in selected_columns:
        if real_data[column].dtype in [np.float64, np.int64]:
            continuous_features.append(column)
        else:
            categorical_features.append(column)

    # Generate synthetic data
    num_samples = len(real_data)
    synthetic_data = []

    fake = Faker()

    for _ in range(num_samples):
        synthetic_sample = {}

        # Generate synthetic values for selected continuous features (within original data range)
        for feature in continuous_features:
            min_value = real_data[feature].min()
            max_value = real_data[feature].max()

            if min_value == max_value:
                synthetic_sample[feature] = min_value
            else:
                range_value = max_value - min_value
                generated_value = np.random.uniform(0, 1) * range_value + min_value
                synthetic_sample[feature] = generated_value

        # Copy selected categorical features from the real data
        for feature in categorical_features:
            synthetic_sample[feature] = real_data[feature]

        synthetic_data.append(synthetic_sample)

    synthetic_data = pd.DataFrame(synthetic_data, columns=selected_columns)

    return synthetic_data

# Function to check if the uploaded file is valid
def is_valid_file(data):
    # Check if the first row contains NULL or NaN values
    has_null_values_in_first_row = data.iloc[0].isnull().any()
    return not has_null_values_in_first_row

def main():
    st.title("Synthetic Data Generator")

    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xls", "xlsx"])
    real_data = None

    if uploaded_file is not None:
        real_data = pd.read_excel(uploaded_file)
        # Check if the uploaded file is valid
        if is_valid_file(real_data):
            st.success("Uploaded file is valid.")
        else:
            st.error("Uploaded file is invalid. The first row contains NULL or NaN values. Please upload a valid file.")
            return

        st.header("Real Data Preview")
        st.write(real_data.head())

        selected_columns = st.multiselect("Select Columns for Synthetic Data", real_data.columns if real_data is not None else [])

        if st.button("Generate Synthetic Data"):
            if not selected_columns:
                st.warning("Please select at least one column for synthetic data generation.")
            elif real_data is not None:
                st.write("Generating synthetic data...")
                synthetic_data = generate_synthetic_data(real_data, selected_columns)

                # Replace the selected columns in the original data with synthetic data
                modified_data = real_data.copy()
                modified_data[selected_columns] = synthetic_data

                # Save the modified data to a new CSV file
                modified_data.to_csv("synthetic_data.csv", index=False)
                st.success("Synthetic data generated and modified data saved to CSV.")

                # Display a preview of the modified data
                st.header("Synthetic Data Preview")
                st.write(modified_data.head())

        if st.button("Download Modified Data"):
            with open("synthetic_data.csv", "rb") as file:
                st.download_button("Download synthetic Data CSV", file.read(), file_name="synthetic_data.csv")

if __name__ == "__main__":
    main()
