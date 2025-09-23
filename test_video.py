#!/usr/bin/env python3
"""
Test runner for findChessboardCorners function.
Usage: python test_findChessboardCorners.py image_path
"""

import sys
import os
import numpy as np
from time import time

# Import the required functions
from chessboard_finder import findChessboardCorners
from helper_image_loading import loadImageGrayscale

def test_findChessboardCorners(image_path):
    """
    Test runner for findChessboardCorners function.
    
    Args:
        image_path (str): Path to the image file to test
        
    Returns:
        tuple: (corners, processing_time) where corners is the result from findChessboardCorners
               and processing_time is the time taken in seconds
    """
    print(f"Testing findChessboardCorners with image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return None, None
    
    try:
        # Load image as grayscale PIL image
        print("Loading image...")
        img_pil = loadImageGrayscale(image_path)
        
        if img_pil is None:
            print("Error: Failed to load image.")
            return None, None
            
        # Convert PIL image to numpy array as required by findChessboardCorners
        print("Converting to numpy array...")
        img_arr = np.asarray(img_pil, dtype=np.float32)
        
        print(f"Image shape: {img_arr.shape}")
        print("Processing with findChessboardCorners...")
        
        # Time the function execution
        start_time = time()
        corners = findChessboardCorners(img_arr)
        processing_time = time() - start_time
        
        print(f"Processing completed in {processing_time:.4f} seconds")
        
        # Display results
        if corners is not None:
            print(f"✅ Chessboard corners found: {corners}")
            print(f"   Top-left corner: ({corners[0]}, {corners[1]})")
            print(f"   Bottom-right corner: ({corners[2]}, {corners[3]})")
            
            # Calculate board dimensions
            board_width = corners[2] - corners[0]
            board_height = corners[3] - corners[1]
            print(f"   Board dimensions: {board_width} x {board_height} pixels")
        else:
            print("❌ No chessboard corners detected in the image")
            
        return corners, processing_time
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function to run the test with command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python test_findChessboardCorners.py <image_path>")
        print("Example: python test_findChessboardCorners.py test_chessboard.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    corners, processing_time = test_findChessboardCorners(image_path)
    
    if corners is not None:
        print(f"\n🎯 Test completed successfully!")
        print(f"   Corners: {corners}")
        print(f"   Processing time: {processing_time:.4f}s")
    else:
        print(f"\n❌ Test failed - no corners detected or error occurred")
        sys.exit(1)

if __name__ == '__main__':
    main()