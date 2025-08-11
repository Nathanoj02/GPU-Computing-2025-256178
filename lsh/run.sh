# Takes input dimensions and generates dataset if it doesn't exist, then run bin/lsh with the provided arguments
#!/bin/bash

# If no dimensions are provided, print error
if [ -z "$1" ]; then
    echo "Usage: $0 <dimensions> [num_points]"
    exit 1
fi

# Default number of points
NUM_POINTS=1000

if [ -n "$2" ]; then
    NUM_POINTS=$2
fi

DIMENSIONS=$1

# Check if dataset exists
if [ ! -f "../dataset/points_${DIMENSIONS}d.txt" ]; then
    # Generate dataset
    python3 python/generate_data.py $DIMENSIONS $NUM_POINTS
fi

# Run the LSH binary
./bin/lsh $DIMENSIONS $NUM_POINTS