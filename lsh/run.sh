# Takes input dimensions and generates dataset if it doesn't exist, then run bin/lsh with the provided arguments
#!/bin/bash

# If no dimensions are provided, print error
if [ -z "$1" ]; then
    echo "Usage: $0 <dimensions> <num_hyperplanes> [num_points]"
    exit 1
fi

# If no number of hyperplanes is provided, print error
if [ -z "$2" ]; then
    echo "Usage: $0 <dimensions> <num_hyperplanes> [num_points]"
    exit 1
fi

# Default number of points
NUM_POINTS=1000

# If a third argument is provided, use it as the number of points
if [ -n "$3" ]; then
    NUM_POINTS=$3
fi

DIMENSIONS=$1
NUM_HYPERPLANES=$2

# Check if dataset exists
if [ ! -f "../dataset/points_${DIMENSIONS}d.txt" ]; then
    # Generate dataset
    python3 python/generate_data.py $DIMENSIONS $NUM_POINTS
fi

# Run the LSH binary
./bin/lsh $DIMENSIONS $NUM_HYPERPLANES