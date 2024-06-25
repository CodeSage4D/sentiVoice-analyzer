import pandas as pd

def inspect_and_clean_csv(file_path):
    try:
        # Read the CSV file to inspect its contents
        df = pd.read_csv(file_path, encoding='utf-8')

        # Display the first few rows to inspect the structure
        print("First few rows:")
        print(df.head())

        # Check the structure of the DataFrame
        print("\nDataFrame info:")
        print(df.info())

        # Check for any missing values
        print("\nMissing values:")
        print(df.isnull().sum())

        # Check for unique values in each column
        print("\nUnique values in 'label' column:")
        print(df['label'].unique())

        # Check for data types of columns
        print("\nData types:")
        print(df.dtypes)

        # Check for any specific irregularities in the data that might cause parsing errors
        # For example, you can inspect specific rows or columns where errors occur
        problematic_rows = []

        # Iterate through the DataFrame and identify problematic rows
        for index, row in df.iterrows():
            try:
                # Example check for unexpected number of fields in a row
                if len(row) != 3:
                    problematic_rows.append(index)
            except Exception as e:
                print(f"Error in row {index}: {e}")

        # Print out any identified problematic rows
        if problematic_rows:
            print("\nProblematic rows:")
            for row_index in problematic_rows:
                print(f"Row {row_index}: {df.iloc[row_index]}")

        # Manually correct any identified issues in the CSV file
        # This might involve editing the CSV file directly using a text editor or programmatically

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.ParserError as pe:
        print(f"Error: Unable to parse CSV file '{file_path}'. Details: {pe}")
    except Exception as e:
        print(f"Error loading data: {e}")

# Specify the path to your CSV file
file_path = 'data/Emotions_filtered.csv'

# Call the function to inspect and clean the CSV file
inspect_and_clean_csv(file_path)
