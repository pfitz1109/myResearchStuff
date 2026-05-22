""" Code ripped straight from Google Gemini """

import pandas as pd
import numpy as np

# =============================================================================
# VARIABLES TO CHANGE FOR FUTURE SIMULATIONS
# =============================================================================
input_file = 'mbSol.csv'            # Your simulation output matrix
output_file = 'paraview_ready.csv'  # The resulting Paraview-compatible file

# Spatial Domain boundaries (top row to bottom row)
x_max = 1.0
x_min = -1.0

# Time Domain boundaries (first column to last column)
t_min = 0.0
t_max = 0.5
# =============================================================================

def convert_matrix_for_paraview():
    print(f"Reading {input_file}...")
    
    # Read the data. `delim_whitespace=True` is crucial here because the 
    # numbers in your file are separated by spaces, not commas.
    df = pd.read_csv(input_file, delim_whitespace=True, header=None)
    matrix = df.values
    
    # Python dynamically counts the rows (space) and columns (time)
    num_rows, num_cols = matrix.shape
    print(f"Detected {num_rows} spatial nodes and {num_cols} time steps.")
    
    # Create the uniform coordinate arrays
    # np.linspace calculates the constant dX and dT spacing for you based on the counts
    x_coords = np.linspace(x_max, x_min, num_rows)
    t_coords = np.linspace(t_min, t_max, num_cols)
    
    # Create a 2D meshgrid mapping out the space-time coordinates
    T, X = np.meshgrid(t_coords, x_coords)
    
    # Flatten the 2D arrays into 1D columns for ParaView
    x_flat = X.flatten()
    y_flat = T.flatten()            # We map Time to the Y-axis 
    z_flat = np.zeros_like(x_flat)  # ParaView expects a 3D coordinate system (Z=0)
    value_flat = matrix.flatten()
    
    # Build the Paraview-ready DataFrame
    out_df = pd.DataFrame({
        'X': x_flat,
        'Y': y_flat,
        'Z': z_flat,
        'Value': value_flat
    })
    
    # Save to standard comma-separated format
    out_df.to_csv(output_file, index=False)
    print(f"Success! Formatted data saved to {output_file}.")

if __name__ == "__main__":
    convert_matrix_for_paraview()