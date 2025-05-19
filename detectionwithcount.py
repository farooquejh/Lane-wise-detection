import cv2
import json
import numpy as np
from ultralytics import YOLO

def load_rois(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    lane_polygons = {}
    for lane_name, points in data.get("lanes", {}).items():
        if len(points) >= 3:
            lane_polygons[lane_name] = np.array(points, dtype=np.int32)
    return lane_polygons

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def detect_and_count(image_path, roi_json, model_path='yolov8n.pt'):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return

    # Load lane ROIs
    lane_polygons = load_rois(roi_json)
    lane_counts = {lane: 0 for lane in lane_polygons.keys()}

    # Load YOLOv8 model
    model = YOLO(model_path)
    results = model(image)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center_point = (center_x, center_y)

        # Check which lane the center falls in
        for lane_name, polygon in lane_polygons.items():
            if point_in_polygon(center_point, polygon):
                lane_counts[lane_name] += 1

                # Draw detection box and label
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{model.names[cls_id]} {conf:.2f}"

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                break  # Stop after first match

    # Draw lane boundaries and counts
    for lane_name, polygon in lane_polygons.items():
        cv2.polylines(image, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        M = cv2.moments(polygon)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
           # cv2.putText(image, f"{lane_name}: {lane_counts[lane_name]}", (cx, cy),
                      #  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # Draw detection count summary at bottom
    summary_text = " | ".join([f"{lane}: {lane_counts[lane]}" for lane in sorted(lane_counts.keys())])
    print("xxxxxxxxxxxxxx",summary_text)
    print(lane_counts)
    
    text_size, _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = image.shape[0] - 20  # 20px above bottom

    cv2.rectangle(image, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
    cv2.putText(image, summary_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # Show result
    cv2.imshow("Lane-wise Detection Count", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_count("image.jpeg", "road_lanes_roi.json", "yolov8n.pt")
