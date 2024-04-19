import pandas as pd

def append_row_to_csv(filename, data):
    # Load the existing data from the CSV file
    df = pd.read_csv(filename)
    
    # Create a new DataFrame with the new row of data
    new_row = pd.DataFrame([data], columns=df.columns)
    
    # Append the new row to the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(filename, index=False)

# Define the filename and the new row data
filename = 'data/top-50.csv'
new_data = [1, 'Doge', 'Doge']

# Call the function to append the new row to the file
append_row_to_csv(filename, new_data)
