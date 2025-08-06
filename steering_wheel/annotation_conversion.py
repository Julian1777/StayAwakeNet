import os
import json
import glob
import shutil
import cv2
import numpy as np

# Base directory structure
base_dir = 'dataset'

# OpenLabel/DriveNet dataset paths - original data
video_dir = os.path.join(base_dir, 'original', 'videos')
anno_dir = os.path.join(base_dir, 'original', 'annotations')

# Output paths for YOLO format data - processed data
output_img_dir = os.path.join(base_dir, 'images')
output_label_dir = os.path.join(base_dir, 'labels')
output_train_val_dir = os.path.join(base_dir, 'splits')

# Create directories
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)
os.makedirs(output_train_val_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
os.makedirs(anno_dir, exist_ok=True)

# Define class mapping for YOLO format
# Focus on microsleep, hands on wheel, and distractions
class_map = {
    "microsleep": 0,              # Microsleep detection
    "hands_both": 1,              # Both hands on wheel
    "hands_one": 2,               # One hand on wheel 
    "hands_none": 3,              # No hands on wheel
    "looking_away": 4,            # Not looking at road
    "phone_usage": 5,             # Using phone (talking/texting)
}

# Set maximum samples per behavior class
max_samples_per_class = 400  # 400 images per class for good balance of accuracy and training time

# Initialize counters for each behavior
behavior_counters = {behavior: 0 for behavior in class_map.keys()}

# Track the total number of images
total_images_captured = 0
max_total_images = 2500  # Total images across all behaviors (increased to accommodate 400 per class)

# Find all JSON annotation files
anno_files = sorted(glob.glob(os.path.join(anno_dir, '*.json')))

# Process each annotation file
for file_idx, anno_path in enumerate(anno_files, 1):
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    # Extract metadata about the video
    video_name = data['openlabel']['metadata']['name']
    total_frames = data['openlabel'].get('frame_intervals', [{'frame_end': 0}])[0].get('frame_end', 0)
    
    # Get video streams information
    streams = data['openlabel'].get('streams', {})
    face_camera = streams.get('face_camera', {}).get('uri', '')
    hands_camera = streams.get('hands_camera', {}).get('uri', '')
    
    print(f"Processing {video_name} with {total_frames} frames")
    
    # Get the actions from the JSON file
    actions = data['openlabel'].get('actions', {})
    
    # Extract frame ranges for each behavior we're interested in
    behavior_frames = {
        "microsleep": [],           # Look for microsleep or sleepy driving frames
        "hands_both": [],           # Both hands on wheel
        "hands_one": [],            # One hand on wheel
        "hands_none": [],           # No hands on wheel
        "looking_away": [],         # Not looking at road
        "phone_usage": []           # Using phone
    }
    
    # Map behavior types from the dataset to our classification
    for action_id, action in actions.items():
        action_type = action.get('type', '')
        
        # Extract frame intervals for this action
        intervals = action.get('frame_intervals', [])
        
        # Map the action types to our behavior categories
        if "microsleep" in action_type or "fatigue" in action_type or "sleepy" in action_type:
            for interval in intervals:
                start = interval.get('frame_start', 0)
                end = interval.get('frame_end', 0)
                print(f"Found {action_type} behavior in frames {start}-{end}")
                behavior_frames["microsleep"].extend(list(range(start, end + 1)))
        
        elif "hands_using_wheel/both" in action_type:
            for interval in intervals:
                start = interval.get('frame_start', 0)
                end = interval.get('frame_end', 0)
                behavior_frames["hands_both"].extend(list(range(start, end + 1)))
                
        elif "hands_using_wheel/only_left" in action_type or "hands_using_wheel/only_right" in action_type:
            for interval in intervals:
                start = interval.get('frame_start', 0)
                end = interval.get('frame_end', 0)
                behavior_frames["hands_one"].extend(list(range(start, end + 1)))
                
        elif "hands_using_wheel/none" in action_type:
            for interval in intervals:
                start = interval.get('frame_start', 0)
                end = interval.get('frame_end', 0)
                behavior_frames["hands_none"].extend(list(range(start, end + 1)))
                
        elif "gaze_on_road/not_looking_road" in action_type:
            for interval in intervals:
                start = interval.get('frame_start', 0)
                end = interval.get('frame_end', 0)
                behavior_frames["looking_away"].extend(list(range(start, end + 1)))
                
        elif "phone" in action_type or "texting" in action_type or "talking_phone" in action_type:
            for interval in intervals:
                start = interval.get('frame_start', 0)
                end = interval.get('frame_end', 0)
                behavior_frames["phone_usage"].extend(list(range(start, end + 1)))
    
    # Process the videos to extract frames and create YOLO annotations
    # Assume face_camera_path and hands_camera_path are paths to the video files
    face_camera_path = os.path.join(video_dir, face_camera)
    hands_camera_path = os.path.join(video_dir, hands_camera)
    
    # Try to open face and hands camera videos
    try:
        face_cap = cv2.VideoCapture(face_camera_path) if os.path.exists(face_camera_path) else None
        hands_cap = cv2.VideoCapture(hands_camera_path) if os.path.exists(hands_camera_path) else None
        
        if face_cap is None and hands_cap is None:
            print(f"Warning: Could not open video files for {video_name}")
            continue
            
        # Get video properties
        width = int(face_cap.get(cv2.CAP_PROP_FRAME_WIDTH) if face_cap else 
                    hands_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(face_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if face_cap else 
                     hands_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = face_cap.get(cv2.CAP_PROP_FPS) if face_cap else hands_cap.get(cv2.CAP_PROP_FPS)
        
        # Process frames at an interval (e.g., every 5 frames)
        # Reduced from 30 to 5 to ensure we don't miss short behaviors
        frame_interval = 5  # Smaller interval to catch more behaviors
        
        # Set a cap on how many images to capture from each video
        max_video_images = 500  # Increased to accommodate more samples per class
        
        # Counter for images captured from this video
        images_captured = 0
        # Global counter for naming across all videos
        global_img_counter = len(glob.glob(os.path.join(output_img_dir, "image_*.jpg")))
        
        # Process frames from the videos
        frame_count = 0
        while True:
            # Check if we've reached the total images cap
            if total_images_captured >= max_total_images:
                print(f"Reached max total images: {max_total_images}")
                break
            
            # Read frames from both cameras
            face_ret, face_frame = face_cap.read() if face_cap else (False, None)
            hands_ret, hands_frame = hands_cap.read() if hands_cap else (False, None)
            
            if not face_ret and not hands_ret:
                break
                
            if frame_count % frame_interval == 0:
                # Check if we've reached the cap for this video
                if images_captured >= max_video_images:
                    break
                
                # Check which behaviors are active in this frame
                active_behaviors = []
                for behavior, frames in behavior_frames.items():
                    if frame_count in frames:
                        active_behaviors.append(behavior)
                
                # Always process frames with behaviors to ensure we don't miss any annotations
                # Skip frames without any interesting behaviors if we already have enough "default" images
                if not active_behaviors and total_images_captured > 100:
                    frame_count += 1
                    continue
                    
                # Process all frames with behaviors, regardless of frame_interval
                # This ensures we don't miss important annotations
                has_important_behavior = False
                for behavior in ["microsleep", "phone_usage", "hands_none"]:  # Priority behaviors
                    if behavior in active_behaviors:
                        has_important_behavior = True
                        break
                
                # Check if we already have enough samples for all behaviors in this frame
                should_skip = True
                for behavior in active_behaviors:
                    if behavior_counters[behavior] < max_samples_per_class:
                        should_skip = False
                        break
                
                # Skip if we have enough samples of all active behaviors (unless it's an important behavior)
                if should_skip and active_behaviors and total_images_captured > 300 and not has_important_behavior:
                    frame_count += 1
                    continue
                
                # Create a combined frame if both are available
                if face_frame is not None and hands_frame is not None:
                    # Stack the images vertically
                    combined_frame = np.vstack((face_frame, hands_frame))
                elif face_frame is not None:
                    combined_frame = face_frame
                else:
                    combined_frame = hands_frame
                
                # Save the frame as an image with sequential numbering
                global_img_counter += 1
                img_name = f"image_{global_img_counter:03d}.jpg"
                img_path = os.path.join(output_img_dir, img_name)
                cv2.imwrite(img_path, combined_frame)
                
                # Create YOLO annotation for this frame with matching name
                label_name = f"anno_{global_img_counter:03d}.txt"
                label_path = os.path.join(output_label_dir, label_name)
                
                images_captured += 1
                
                with open(label_path, "w") as label_file:
                    # Check which behaviors are active in this frame
                    labels = []
                    frame_behaviors = []
                    
                    for behavior, frames in behavior_frames.items():
                        if frame_count in frames:
                            class_id = class_map[behavior]
                            # Update counter for this behavior
                            behavior_counters[behavior] += 1
                            frame_behaviors.append(behavior)
                            
                            # For simple classification without bounding box,
                            # use the entire image with class ID
                            # Format: class_id, x_center, y_center, width, height
                            # All normalized to [0-1]
                            labels.append(f"{class_id} 0.5 0.5 1.0 1.0")
                    
                    # Write all labels for this frame
                    if labels:
                        label_file.write("\n".join(labels))
                        
                        # Print status message
                        if len(frame_behaviors) > 0:
                            print(f"Added frame {global_img_counter} with behaviors: {', '.join(frame_behaviors)}")
                    else:
                        # Create an empty file (no annotations) - this is a "normal" frame
                        print(f"Added frame {global_img_counter} with no specific behaviors")
                    
                    # Update total images counter
                    total_images_captured += 1
            
            frame_count += 1
        
        # Clean up resources
        if face_cap:
            face_cap.release()
        if hands_cap:
            hands_cap.release()
            
    except Exception as e:
        print(f"Error processing {video_name}: {str(e)}")
        continue

# Create train/val split
all_images = sorted(glob.glob(os.path.join(output_img_dir, "image_*.jpg")))
val_split = 0.2  # 20% for validation
val_count = int(len(all_images) * val_split)

# Shuffle the list
import random
random.seed(42)  # For reproducibility
random.shuffle(all_images)

# Create train.txt and val.txt
with open(os.path.join(output_train_val_dir, "train.txt"), "w") as f:
    for img_path in all_images[val_count:]:
        f.write(f"{os.path.abspath(img_path)}\n")
        
with open(os.path.join(output_train_val_dir, "val.txt"), "w") as f:
    for img_path in all_images[:val_count]:
        f.write(f"{os.path.abspath(img_path)}\n")

# Print statistics on behavior distribution
print("\nDataset Statistics:")
print(f"Total images: {total_images_captured}")
for behavior, count in behavior_counters.items():
    percentage = (count / total_images_captured) * 100 if total_images_captured > 0 else 0
    print(f"- {behavior}: {count} images ({percentage:.1f}%)")

print(f"\nConversion complete. {len(all_images)} images processed.")
print(f"Data saved to {base_dir}")

# Create dataset.yaml for YOLO training
yaml_content = f"""
path: {os.path.abspath(base_dir)}
train: {os.path.join('splits', 'train.txt')}
val: {os.path.join('splits', 'val.txt')}

nc: {len(class_map)}
names: {list(class_map.keys())}
"""

with open(os.path.join(base_dir, "dataset.yaml"), "w") as f:
    f.write(yaml_content)

print(f"Created dataset.yaml for YOLO training")