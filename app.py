import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def validate_bbox(x1, y1, x2, y2, frame_width, frame_height):
    """Validate and clamp bounding box coordinates"""
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))
    return int(x1), int(y1), int(x2), int(y2)

# === Load YOLOv11 Model ===
model_path = 'best.pt'  # Your fine-tuned YOLOv11 model

print("🔄 Loading YOLOv11 model...")
model = YOLO(model_path)

# Get class names from the model
names = model.names
print(f"✅ All classes: {names}")

# Identify the class index for "player" (case-insensitive)
player_class_ids = []
if isinstance(names, dict):
    player_class_ids = [i for i, name in names.items() if "player" in str(name).lower()]
else:
    # If names is a list
    player_class_ids = [i for i, name in enumerate(names) if "player" in str(name).lower()]

print(f"✅ Player class index(es): {player_class_ids}")

# If no "player" class found, use "person" class (COCO class 0)
if not player_class_ids:
    print("⚠️ No 'player' class found, looking for 'person' class...")
    if isinstance(names, dict):
        person_class_ids = [i for i, name in names.items() if "person" in str(name).lower()]
    else:
        person_class_ids = [i for i, name in enumerate(names) if "person" in str(name).lower()]
    
    if person_class_ids:
        player_class_ids = person_class_ids
        print(f"✅ Using 'person' class index(es): {player_class_ids}")
    else:
        print("❌ No suitable class found for player detection")
        exit(1)

# === Deep SORT Tracker ===
tracker = DeepSort(
    max_iou_distance=0.7, 
    max_age=30, 
    n_init=3, 
    nn_budget=100,
    max_cosine_distance=0.4
)

# === Input/output paths ===
input_path = "input_video.mp4"
output_path = "output_video.mp4"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("❌ Error: Could not open video file")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"📹 Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    
    print(f"🔄 Processing frame {frame_count}/{total_frames}", end='\r')

    # === YOLOv11 Inference ===
    results = model(frame, conf=0.25, iou=0.5, verbose=False)
    
    detections = []
    
    # Process YOLOv11 results
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Validate and clamp coordinates
                x1, y1, x2, y2 = validate_bbox(int(x1), int(y1), int(x2), int(y2), width, height)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                label = names[class_id] if class_id < len(names) else f'class {class_id}'

                # Draw all detection boxes (for visual debug)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 1)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

                # Add to tracker only if it's a player
                if class_id in player_class_ids:
                    # Convert to [x, y, w, h] format for DeepSort
                    w, h = x2 - x1, y2 - y1
                    
                    # Additional validation
                    if w > 10 and h > 10:  # Minimum size threshold
                        detections.append(([x1, y1, w, h], confidence, None))

    # === Track players ===
    tracks = tracker.update_tracks(detections, frame=frame)
    
    tracked_count = 0
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        ltrb = track.to_ltrb()
        
        # Validate track coordinates
        x1, y1, x2, y2 = validate_bbox(*map(int, ltrb), width, height)
        
        # Skip invalid tracks
        if x2 <= x1 or y2 <= y1:
            continue
            
        tracked_count += 1
        
        # Draw tracking box (green)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw track ID
        cv2.putText(frame, f"Player ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)

    # Display frame info
    info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Tracked: {tracked_count}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)

    out.write(frame)

print(f"\n✅ Processing complete!")
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Output saved to: {output_path}")