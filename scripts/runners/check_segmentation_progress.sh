#!/bin/bash
# Script to check the progress of the segmentation process

PROGRESS_FILE="results/segmentation_output/segmentation_progress.txt"

# Check if the progress file exists
if [ -f "$PROGRESS_FILE" ]; then
    echo "Segmentation Progress:"
    echo "----------------------"
    cat "$PROGRESS_FILE"
    
    # Get CPU usage of the process
    PID=$(pgrep -f "python run_parallel_segmentation.py")
    if [ -n "$PID" ]; then
        echo ""
        echo "Process Stats:"
        echo "--------------"
        echo "PID: $PID"
        echo "CPU Usage: $(ps -p $PID -o %cpu | tail -n 1)%"
        echo "Memory Usage: $(ps -p $PID -o %mem | tail -n 1)%"
        echo "Running for: $(ps -p $PID -o etime | tail -n 1)"
        
        # Show the top 5 most CPU-intensive threads
        echo ""
        echo "Top Worker Threads:"
        echo "-----------------"
        ps -M $PID -o %cpu,command | grep Python | sort -rn | head -5
    else
        echo ""
        echo "Segmentation process is not currently running."
    fi
    
    # Show recent completed images
    echo ""
    echo "Recently Completed Images:"
    echo "-------------------------"
    grep "Completed" results/segmentation_output/segmentation.log | tail -5
else
    echo "Progress file not found. Segmentation may not have started yet."
    echo "Check if process is running with: ps aux | grep run_parallel_segmentation"
fi