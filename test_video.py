import os
import cv2
import numpy as np
import argparse
import json
import PIL.Image
from video_helpers import VideoContainer
from chessboard_finder import findChessboardCorners

def extract_frames(video_path, output_dir="frames", frame_format="png", interval_seconds=0.5, diff_threshold=500, denoise=True, roi=None, auto_detect_chessboard=False):
    """
    Extract frames from an MP4 video at specified time intervals and save as individual images.
    Only saves frames that are significantly different from the previous saved frame.
    
    Args:
        video_path (str): Path to the input MP4 video file
        output_dir (str): Directory to save extracted frames (default: "frames")
        frame_format (str): Image format for saved frames (default: "png")
        interval_seconds (float): Time interval between extracted frames in seconds (default: 0.5)
        diff_threshold (int): Minimum sum of absolute differences to consider frames different (default: 500)
        denoise (bool): Apply denoising to reduce false positives from noise (default: True)
        roi (tuple): Region of interest as (x, y, width, height) to focus analysis and output (default: None for full frame)
        auto_detect_chessboard (bool): Automatically detect chessboard in each frame and use as ROI (default: False)
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Initialize video container
    try:
        video = VideoContainer(video_path)
        print(f"Video loaded successfully:")
        print(f"  - Total frames: {video.frame_count}")
        print(f"  - Frame rate: {video.frame_rate:.2f} fps")
        print(f"  - Resolution: {video.frame_width}x{video.frame_height}")
        print(f"  - Duration: {video.frame_count/video.frame_rate:.2f} seconds")
        print(f"  - Extracting every {interval_seconds} seconds")
        print(f"  - Difference threshold: {diff_threshold}")
        print(f"  - Denoising enabled: {denoise}")
        print(f"  - Auto-detect chessboard: {auto_detect_chessboard}")
        if roi and not auto_detect_chessboard:
            print(f"  - Region of interest: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
            # Validate ROI bounds
            if (roi[0] < 0 or roi[1] < 0 or 
                roi[0] + roi[2] > video.frame_width or 
                roi[1] + roi[3] > video.frame_height):
                print(f"Error: ROI extends beyond frame boundaries")
                return
        elif not auto_detect_chessboard:
            print(f"  - Region of interest: Full frame")
    except Exception as e:
        print(f"Error loading video: {e}")
        return
    
    # Calculate frame interval based on time interval
    frame_interval = int(video.frame_rate * interval_seconds)
    total_duration = video.frame_count / video.frame_rate
    estimated_frames = int(total_duration / interval_seconds) + 1
    
    print(f"  - Frame interval: every {frame_interval} frames")
    print(f"  - Estimated output frames: {estimated_frames}")
    
    # Extract and save frames at intervals
    extracted_count = 0
    skipped_count = 0
    chessboard_not_found_count = 0
    previous_frame_gray = None
    last_detected_roi = None  # Cache last successful chessboard detection
    extracted_timestamps = []  # Store timestamps of extracted frames
    
    try:
        # Start from frame 0
        video.seek_to(0)
        
        frame_num = 0
        while frame_num < video.frame_count:
            # Seek to the specific frame
            video.seek_to(frame_num)
            
            # Read the current frame
            frame = video.read()
            
            if frame is None:
                print(f"Warning: Could not read frame {frame_num}")
                break
            
            # Determine ROI for this frame
            current_roi = roi
            frame_for_comparison = frame
            
            if auto_detect_chessboard:
                # Convert frame to grayscale for chessboard detection
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                
                # Detect chessboard corners
                corners = findChessboardCorners(frame_gray)
                
                if corners is not None:
                    # Convert corners to ROI format (x, y, width, height)
                    x, y, x2, y2 = corners
                    current_roi = (int(x), int(y), int(x2 - x), int(y2 - y))
                    last_detected_roi = current_roi  # Cache for next frame
                    
                    # Extract chessboard region
                    frame_for_comparison = frame[y:y2, x:x2]
                    
                    print(f"Frame {frame_num}: Chessboard detected at ROI: x={current_roi[0]}, y={current_roi[1]}, w={current_roi[2]}, h={current_roi[3]}")
                
                elif last_detected_roi is not None:
                    # Use last detected ROI if current detection failed
                    current_roi = last_detected_roi
                    x, y, w, h = current_roi
                    
                    # Ensure ROI is within frame bounds
                    if (x >= 0 and y >= 0 and 
                        x + w <= frame.shape[1] and 
                        y + h <= frame.shape[0]):
                        frame_for_comparison = frame[y:y+h, x:x+w]
                        print(f"Frame {frame_num}: Using cached chessboard ROI: x={x}, y={y}, w={w}, h={h}")
                    else:
                        print(f"Frame {frame_num}: Cached ROI out of bounds, using full frame")
                        frame_for_comparison = frame
                        current_roi = None
                
                else:
                    # No chessboard detected and no cached ROI
                    chessboard_not_found_count += 1
                    print(f"Frame {frame_num}: No chessboard detected, skipping frame")
                    frame_num += frame_interval
                    continue
            
            elif roi:
                # Use manual ROI
                x, y, w, h = roi
                frame_for_comparison = frame[y:y+h, x:x+w]
                current_roi = roi
            
            # Convert current frame to grayscale for comparison
            current_frame_gray = cv2.cvtColor(frame_for_comparison, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising if enabled
            if denoise:
                current_frame_gray = denoise_frame(current_frame_gray)
            
            # Check if this frame is different enough from the previous saved frame
            should_save = True
            diff_sum = 0
            
            if previous_frame_gray is not None:
                # Ensure frames are the same size for comparison
                if current_frame_gray.shape == previous_frame_gray.shape:
                    # Calculate absolute difference between current and previous frame
                    diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
                    
                    # Apply additional morphological operations to reduce noise in difference
                    if denoise:
                        diff = denoise_difference(diff)
                    
                    diff_sum = np.sum(diff)
                    
                    # Only save if difference is above threshold
                    should_save = diff_sum > diff_threshold
                else:
                    # If frame sizes don't match (different ROI), always save
                    print(f"Frame {frame_num}: Frame size changed, saving frame")
                    should_save = True
            
            if should_save:
                # Calculate timestamp for this frame (in seconds and milliseconds)
                timestamp_seconds = frame_num / video.frame_rate
                timestamp_milliseconds = int(timestamp_seconds * 1000)
                
                # Generate filename with timestamp
                if auto_detect_chessboard:
                    filename = f"frame_{extracted_count:06d}_t{timestamp_seconds:.1f}s_chess.{frame_format}"
                elif current_roi:
                    filename = f"frame_{extracted_count:06d}_t{timestamp_seconds:.1f}s_roi.{frame_format}"
                else:
                    filename = f"frame_{extracted_count:06d}_t{timestamp_seconds:.1f}s.{frame_format}"
                filepath = os.path.join(output_dir, filename)
                
                # Save the frame (chessboard/ROI if specified, otherwise full frame)
                success = cv2.imwrite(filepath, frame_for_comparison)
                
                if success:
                    extracted_count += 1
                    # Add timestamp to the list
                    extracted_timestamps.append(timestamp_milliseconds)
                    
                    if auto_detect_chessboard and current_roi:
                        roi_info = f" (Chessboard: {current_roi[2]}x{current_roi[3]})"
                    elif current_roi:
                        roi_info = f" (ROI: {current_roi[2]}x{current_roi[3]})"
                    else:
                        roi_info = ""
                    print(f"Extracted frame {extracted_count}: {filename} (frame #{frame_num}, diff: {diff_sum:.0f}, timestamp: {timestamp_milliseconds}ms){roi_info}")
                    # Update previous frame for next comparison
                    previous_frame_gray = current_frame_gray.copy()
                else:
                    print(f"Error: Could not save frame {frame_num} to {filepath}")
            else:
                skipped_count += 1
                print(f"Skipped frame {frame_num} (diff: {diff_sum:.0f} < {diff_threshold})")
            
            # Move to next interval
            frame_num += frame_interval
        
        # Save timestamps to JSON file
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        json_filename = f"{video_filename}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        # Create JSON data structure
        json_data = {
            "video_file": os.path.basename(video_path),
            "video_path": video_path,
            "extraction_settings": {
                "interval_seconds": interval_seconds,
                "diff_threshold": diff_threshold,
                "denoise_enabled": denoise,
                "auto_chessboard": auto_detect_chessboard,
                "roi": roi,
                "frame_format": frame_format
            },
            "video_info": {
                "total_frames": video.frame_count,
                "frame_rate": video.frame_rate,
                "duration_seconds": video.frame_count / video.frame_rate,
                "resolution": {
                    "width": video.frame_width,
                    "height": video.frame_height
                }
            },
            "extraction_results": {
                "frames_extracted": extracted_count,
                "frames_skipped": skipped_count,
                "frames_no_chessboard": chessboard_not_found_count if auto_detect_chessboard else 0,
                "total_frames_processed": extracted_count + skipped_count + chessboard_not_found_count
            },
            "timestamps_milliseconds": extracted_timestamps
        }
        
        try:
            with open(json_filepath, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, indent=2, ensure_ascii=False)
            print(f"📄 Timestamps saved to: {json_filename}")
        except Exception as json_error:
            print(f"⚠️  Warning: Failed to save JSON file: {str(json_error)}")
        
        print(f"\nExtraction complete!")
        print(f"Successfully extracted {extracted_count} frames to '{output_dir}' directory")
        print(f"Skipped {skipped_count} similar frames")
        if auto_detect_chessboard:
            print(f"Skipped {chessboard_not_found_count} frames due to no chessboard detection")
        print(f"Frames extracted every {interval_seconds} seconds with difference threshold {diff_threshold}")
        if auto_detect_chessboard:
            print(f"Frames automatically cropped to detected chessboard regions")
        elif roi:
            print(f"Frames cropped to ROI: {roi[2]}x{roi[3]} pixels")
        print(f"Timestamps saved to: {json_filename} ({len(extracted_timestamps)} timestamps)")
        
    except Exception as e:
        print(f"Error during frame extraction: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        video._cap.release()

def denoise_frame(frame_gray):
    """
    Apply denoising to a grayscale frame to reduce noise before comparison.
    
    Args:
        frame_gray (numpy.ndarray): Grayscale frame
        
    Returns:
        numpy.ndarray: Denoised grayscale frame
    """
    # Apply Gaussian blur to reduce high-frequency noise
    denoised = cv2.GaussianBlur(frame_gray, (3, 3), 0)
    
    # Optional: Apply Non-local Means Denoising for better results (slower)
    # Uncomment the line below for better denoising at the cost of performance
    # denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
    
    return denoised

def denoise_difference(diff):
    """
    Apply morphological operations to clean up the difference image,
    removing small noise artifacts while preserving significant changes.
    
    Args:
        diff (numpy.ndarray): Difference image from cv2.absdiff
        
    Returns:
        numpy.ndarray: Cleaned difference image
    """
    # Apply a small threshold to eliminate very small differences (likely noise)
    _, diff_thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Opening: erosion followed by dilation (removes small noise)
    diff_opened = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
    
    # Closing: dilation followed by erosion (fills small gaps)
    diff_cleaned = cv2.morphologyEx(diff_opened, cv2.MORPH_CLOSE, kernel)
    
    # Apply the cleaned mask to the original difference
    result = cv2.bitwise_and(diff, diff, mask=diff_cleaned)
    
    return result

def parse_roi(roi_string):
    """
    Parse ROI string in format "x,y,width,height" into tuple of integers.
    
    Args:
        roi_string (str): ROI in format "x,y,width,height"
        
    Returns:
        tuple: (x, y, width, height) as integers
        
    Raises:
        ValueError: If format is invalid
    """
    try:
        parts = roi_string.split(',')
        if len(parts) != 4:
            raise ValueError("ROI must have exactly 4 values")
        
        x, y, w, h = map(int, parts)
        
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError("ROI values must be non-negative, width and height must be positive")
        
        return (x, y, w, h)
    
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid ROI format. Expected 'x,y,width,height', got '{roi_string}': {e}")

def main():
    """Main function to handle command line arguments and run frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract frames from an MP4 video at specified time intervals, only saving frames that differ significantly from the previous saved frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_video.py video.mp4
  python test_video.py video.mp4 -t 1000 -i 1.0
  python test_video.py video.mp4 --roi 100,50,400,400
  python test_video.py video.mp4 --auto-chessboard -o chess_moves
  python test_video.py video.mp4 --auto-chessboard -t 300 -i 2.0

Output:
  - Extracted frames saved as images in the output directory
  - JSON file (videoname.json) with timestamps in milliseconds of extracted frames
        """
    )
    parser.add_argument(
        "input_video", 
        help="Path to the input MP4 video file"
    )
    parser.add_argument(
        "-o", "--output", 
        default="frames",
        help="Output directory for extracted frames (default: frames)"
    )
    parser.add_argument(
        "-f", "--format",
        default="png", 
        choices=["png", "jpg", "jpeg", "bmp"],
        help="Image format for saved frames (default: png)"
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=0.5,
        help="Time interval between extracted frames in seconds (default: 0.5)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=500,
        help="Minimum sum of absolute differences to consider frames different (default: 500)"
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable denoising (may result in more false positives)"
    )
    parser.add_argument(
        "--roi",
        type=str,
        help="Region of interest as 'x,y,width,height' (e.g., '100,50,400,300'). Only this area will be analyzed for differences and saved."
    )
    parser.add_argument(
        "--auto-chessboard",
        action="store_true",
        help="Automatically detect chessboard in each frame and use as ROI. This will only save the chessboard portion when changes are detected."
    )
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.input_video):
        print(f"Error: Video file '{args.input_video}' not found")
        return
    
    # Check if input is a video file
    if not args.input_video.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print(f"Warning: '{args.input_video}' may not be a supported video format")
    
    # Validate interval
    if args.interval <= 0:
        print(f"Error: Interval must be greater than 0 seconds")
        return
    
    # Validate threshold
    if args.threshold < 0:
        print(f"Error: Threshold must be non-negative")
        return
    
    # Check for conflicting options
    if args.roi and args.auto_chessboard:
        print(f"Error: Cannot use both --roi and --auto-chessboard options together")
        return
    
    # Parse ROI if provided
    roi = None
    if args.roi:
        try:
            roi = parse_roi(args.roi)
            print(f"Using ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")
        except ValueError as e:
            print(f"Error: {e}")
            return
    
    if args.auto_chessboard:
        print(f"Auto-chessboard detection enabled - will detect and extract chessboard regions")
    
    print(f"Starting frame extraction from: {args.input_video}")
    extract_frames(args.input_video, args.output, args.format, args.interval, args.threshold, not args.no_denoise, roi, args.auto_chessboard)

if __name__ == "__main__":
    main()