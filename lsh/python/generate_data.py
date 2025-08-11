import numpy as np
import sys
import os

def generate_data(dimensions, num_points):
    """
    Generates random data points in a specified number of dimensions.
    
    Parameters:
    dimensions (int): The number of dimensions for each data point.
    num_points (int): The number of data points to generate.
    """
    np.random.seed(0)  # For reproducibility
    
    data = np.random.rand(num_points, dimensions)
    
    # Save the data to a file in dataset folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, '..', 'dataset', f'points_{dimensions}d.txt')
    np.savetxt(filename, data)


if __name__ == '__main__':
    # Get dimensions and number of points from command line arguments
    if len(sys.argv) != 3:
        print("Usage: python3 generate_data.py <dimensions> <num_points>")
        sys.exit(1)

    dimensions = int(sys.argv[1])
    num_points = int(sys.argv[2])

    # Generate the data
    generate_data(dimensions, num_points)