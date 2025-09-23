import os
import cv2
import argparse
from video_helpers import VideoContainer

def extract_frames(video_path, output_dir="frames", frame_format="png", interval_seconds=0.5):
    """
    Extract frames from an MP4 video at specified time intervals and save as individual images.
    
    Args:
        video_path (str): Path to the input MP4 video file
        output_dir (str): Directory to save extracted frames (default: "frames")
        frame_format (str): Image format for saved frames (default: "png")
        interval_seconds (float): Time interval between extracted frames in seconds (default: 0.5)
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
            
            # Calculate timestamp for this frame
            timestamp = frame_num / video.frame_rate
            
            # Generate filename with timestamp
            filename = f"frame_{extracted_count:06d}_t{timestamp:.1f}s.{frame_format}"
            filepath = os.path.join(output_dir, filename)
            
            # Save the frame
            success = cv2.imwrite(filepath, frame)
            
            if success:
                extracted_count += 1
                print(f"Extracted frame {extracted_count}: {filename} (frame #{frame_num})")
            else:
                print(f"Error: Could not save frame {frame_num} to {filepath}")
            
            # Move to next interval
            frame_num += frame_interval
        
        print(f"\nExtraction complete!")
        print(f"Successfully extracted {extracted_count} frames to '{output_dir}' directory")
        print(f"Frames extracted every {interval_seconds} seconds")
        
    except Exception as e:
        print(f"Error during frame extraction: {e}")
    
    finally:
        # Clean up
        video._cap.release()

def main():
    """Main function to handle command line arguments and run frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract frames from an MP4 video at specified time intervals"
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
    
    print(f"Starting frame extraction from: {args.input_video}")
    extract_frames(args.input_video, args.output, args.format, args.interval)

if __name__ == "__main__":
    main()