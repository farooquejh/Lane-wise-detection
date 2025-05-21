import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import time

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
                    
                    print(f" Loaded lane ROIs from: {json_path}")
                    return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading saved lanes: {e}")
    return False

def select_lane_rois_from_first_frame(video_path):
    global drawing_image, original_image, current_lane, lanes, complete

    # Try loading saved lanes
    lanes_loaded = load_saved_lanes()

    if lanes_loaded and all(complete.values()):
        # Convert to polygons directly and return
        lane_polygons = {}
        for lane, points in lanes.items():
            if len(points) >= 3:
                lane_polygons[lane] = np.array(points, dtype=np.int32)
        print(" Loaded saved lanes. Skipping ROI drawing.")
        return lane_polygons

    # If no valid saved lanes, start drawing
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(" Failed to read first frame.")
        return None

    original_image = frame.copy()
    drawing_image = frame.copy()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n=== ROI Selection Mode ===")
    print("• Left click = add point")
    print("• Right click = close polygon")
    print("• Keys 1/2/3 = switch lanes")
    print("• Press 'd' = done\n")

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
        elif key == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    # Convert and save
    lane_polygons = {}
    lane_data_for_json = {}
    for lane, points in lanes.items():
        if len(points) >= 3:
            lane_polygons[lane] = np.array(points, dtype=np.int32)
            lane_data_for_json[lane] = points

    with open("saved_lanes.json", "w") as f:
        json.dump({"lanes": lane_data_for_json}, f, indent=4)
  

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

    # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    model = YOLO(model_path)

    frame_count = 0
    last_processed_frame = None
    print_interval = 10  # Print every 10 seconds
    last_print_time = time.time()
    interval_lane_counts = {lane: 0 for lane in lane_polygons}
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()

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
                        if cls_id not in [2,3,5,7]:
                            continue
                        interval_lane_counts[lane] += 1
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
            if current_time - last_print_time >= print_interval:
                # Print only the lane counts in dictionary format
                print(" Count in every10second",interval_lane_counts)
                
                # Reset interval counters
                interval_lane_counts = {lane: 0 for lane in lane_polygons}
                last_print_time = current_time
                print("Counts reset at:", time.strftime("%H:%M:%S"),interval_lane_counts)
            #print(lane_counts)
            # for lane ,count in lane_counts.items():
            #     print(lane,"xx",count)
            last_processed_frame = processed_frame
            cv2.imshow("Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write to output
        

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
   

if __name__ == "__main__":
    video_path = "4.mp4"
    output_path = "output.mp4"
    model_path = "yolov8n_openvino_model/"
    process_every = 5

    lane_polygons = select_lane_rois_from_first_frame(video_path)
    if lane_polygons:
        detect_and_count_video(video_path, output_path, lane_polygons, model_path, process_every)