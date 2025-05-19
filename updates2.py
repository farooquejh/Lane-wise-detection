import cv2
import numpy as np
from ultralytics import YOLO
import json
import os

# Global variables for ROI selection
lanes = {"lane1": [], "lane2": [], "lane3": []}
complete = {"lane1": False, "lane2": False, "lane3": False}
current_lane = "lane1"
drawing_image = None
original_image = None
window_name = "Draw Lane ROIs"

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def mouse_callback(event, x, y, flags, param):
    global drawing_image, original_image, lanes, current_lane, complete

    temp_img = original_image.copy()

    lane_colors = {
        "lane1": (255, 180, 0),
        "lane2": (255, 100, 0),
        "lane3": (255, 0, 0)
    }

    for lane_name, points in lanes.items():
        color = lane_colors[lane_name]
        for i in range(len(points)):
            cv2.circle(temp_img, points[i], 5, color, -1)
            if i > 0:
                cv2.line(temp_img, points[i - 1], points[i], color, 2)
        if complete[lane_name] and len(points) > 2:
            cv2.line(temp_img, points[-1], points[0], color, 2)
        if lane_name == current_lane and len(points) > 0 and not complete[lane_name]:
            cv2.line(temp_img, points[-1], (x, y), color, 1)

    cv2.putText(temp_img, f"Current Lane: {current_lane}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane_colors[current_lane], 2)

    if event == cv2.EVENT_LBUTTONDOWN and not complete[current_lane]:
        lanes[current_lane].append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(lanes[current_lane]) > 2:
        complete[current_lane] = True

    drawing_image = temp_img
    cv2.imshow(window_name, drawing_image)

def load_saved_lanes():
    """Load lane ROIs from saved JSON file if it exists."""
    json_path = "saved_lanes.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                if "lanes" in data:
                    loaded_lanes = data["lanes"]
                    global lanes, complete
                    
                    # Reset lanes and complete dictionaries
                    lanes = {"lane1": [], "lane2": [], "lane3": []}
                    complete = {"lane1": False, "lane2": False, "lane3": False}
                    
                    # Load the saved lanes
                    for lane_name, points in loaded_lanes.items():
                        if lane_name in lanes and len(points) >= 3:
                            # Convert points from list of lists to list of tuples
                            lanes[lane_name] = [tuple(point) for point in points]
                            complete[lane_name] = True
                    
                    print(f"✅ Loaded lane ROIs from: {json_path}")
                    return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading saved lanes: {e}")
    return False

def select_lane_rois_from_first_frame(video_path):
    global drawing_image, original_image, current_lane, lanes, complete

    # First try to load existing lanes
    lanes_loaded = load_saved_lanes()

    # If lanes were loaded successfully and all needed lanes are defined, offer to use them
    if lanes_loaded and all(complete.values()):
        print("\n=== Saved Lane ROIs Found ===")
        print("Do you want to use these saved lanes? (y/n)")
        print("Enter 'y' to use saved lanes as-is")
        print("Enter 'e' to edit saved lanes")
        print("Enter 'n' to draw new lanes from scratch")
        choice = input().lower()
        
        if choice == 'y':
            # Convert to polygons
            lane_polygons = {}
            for lane, points in lanes.items():
                if len(points) >= 3:
                    lane_polygons[lane] = np.array(points, dtype=np.int32)
            return lane_polygons
        elif choice == 'n':
            # Reset lanes if user wants to draw new ones
            lanes = {"lane1": [], "lane2": [], "lane3": []}
            complete = {"lane1": False, "lane2": False, "lane3": False}
            lanes_loaded = False
        # For 'e', we keep the loaded lanes for editing

    # Either no saved lanes, or user wants to redefine them
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read first frame.")
        return None

    original_image = frame.copy()
    drawing_image = frame.copy()

    # If lanes were loaded but user wants to edit them, display the loaded lanes
    if lanes_loaded:
        # Pre-draw the loaded lanes
        for lane_name, points in lanes.items():
            if complete[lane_name]:
                for i in range(len(points)):
                    if i > 0:
                        cv2.line(drawing_image, points[i-1], points[i], (255, 100, 0), 2)
                if len(points) > 2:
                    cv2.line(drawing_image, points[-1], points[0], (255, 100, 0), 2)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n=== ROI Selection Instructions ===")
    print("• Left click to draw polygon points.")
    print("• Right click to close current polygon.")
    print("• Press 1, 2, 3 to switch between lanes.")
    print("• Press 'd' to finish ROI selection.\n")

    while True:
        cv2.imshow(window_name, drawing_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            current_lane = "lane1"
            print("Switched to lane1")
        elif key == ord('2'):
            current_lane = "lane2"
            print("Switched to lane2")
        elif key == ord('3'):
            current_lane = "lane3"
            print("Switched to lane3")
        elif key == ord('d'):
            break
        elif key == 27:  # ESC key
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    # Convert to polygons
    lane_polygons = {}
    lane_data_for_json = {}
    for lane, points in lanes.items():
        if len(points) >= 3:
            lane_polygons[lane] = np.array(points, dtype=np.int32)
            lane_data_for_json[lane] = points
    
    # Save the lanes to JSON
    json_path = "saved_lanes.json"
    with open(json_path, "w") as f:
        json.dump({"lanes": lane_data_for_json}, f, indent=4)
    print(f"✅ Lane ROIs saved to: {json_path}")
    return lane_polygons

def detect_and_count_video(video_path, output_path, lane_polygons, model_path='yolov8n.pt', process_every=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Width:", width)
    print("Height:", height)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    model = YOLO(model_path)

    frame_count = 0
    last_processed_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % process_every == 0:
            lane_counts = {lane: 0 for lane in lane_polygons}
            processed_frame = frame.copy()
            results = model(processed_frame)[0]

            # Draw lane polygons
            for lane, poly in lane_polygons.items():
                cv2.polylines(processed_frame, [poly], isClosed=True, color=(255, 0, 0), thickness=2)

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                center = (cx, cy)

                for lane, poly in lane_polygons.items():
                    if point_in_polygon(center, poly):
                        lane_counts[lane] += 1
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = f"{model.names[cls_id]} {conf:.2f}"

                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(processed_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        break

            # Display lane counts
            for lane, poly in lane_polygons.items():
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(processed_frame, f"{lane}: {lane_counts[lane]}", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            summary_text = " | ".join([f"{lane}: {lane_counts[lane]}" for lane in sorted(lane_counts.keys())])
            print(lane_counts)
            
            text_size, _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = (width - text_size[0]) // 2
            text_y = height - 20  # 20px above bottom

            cv2.rectangle(processed_frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            cv2.putText(processed_frame, summary_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            last_processed_frame = processed_frame
            cv2.imshow("Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write to output
        if last_processed_frame is not None:
            out.write(last_processed_frame)
        else:
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video saved: {output_path}")

if __name__ == "__main__":
    video_path = "video.dav"
    output_path = "output.mp4"
    model_path = "yolov8n.pt"
    process_every = 5

    lane_polygons = select_lane_rois_from_first_frame(video_path)
    if lane_polygons:
        detect_and_count_video(video_path, output_path, lane_polygons, model_path, process_every)