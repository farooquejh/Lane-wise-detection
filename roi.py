import cv2
import numpy as np
import json

# Global variables
image = None
original_image = None
lanes = {"lane1": [], "lane2": [], "lane3": []}
current_lane = "lane1"
complete = {"lane1": False, "lane2": False, "lane3": False}
window_name = 'Road Lane ROIs'

def mouse_callback(event, x, y, flags, param):
    global image, original_image, lanes, current_lane, complete
    
    # Make a copy of the original image for drawing
    temp_img = original_image.copy()
    
    # Blue color scheme for lanes (light blue, medium blue, dark blue)
    lane_colors = {
        "lane1": (255, 0, 0),    # Light blue (BGR format)
        "lane2": (255, 0, 0),    # Medium blue
        "lane3": (255, 0, 0)       # Dark blue
    }
    
    # Draw all lanes that have points
    for lane_name, points in lanes.items():
        color = lane_colors[lane_name]
        
        # Draw existing points and lines for all lanes
        if len(points) > 0:
            for i in range(len(points)):
                cv2.circle(temp_img, points[i], 5, color, -1)
                if i > 0:
                    cv2.line(temp_img, points[i-1], points[i], color, 2)
            
            # Draw closing line if lane is complete
            if complete[lane_name] and len(points) > 2:
                cv2.line(temp_img, points[-1], points[0], color, 2)
        
        # Draw line from last point to mouse position for current lane
        if lane_name == current_lane and len(points) > 0 and not complete[lane_name]:
            cv2.line(temp_img, points[-1], (x, y), lane_colors[current_lane], 2)
    
    # Add text showing current lane
    cv2.putText(temp_img, f"Current Lane: {current_lane}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane_colors[current_lane], 2)
    
    # Left button click to add a point to current lane
    if event == cv2.EVENT_LBUTTONDOWN and not complete[current_lane]:
        lanes[current_lane].append((x, y))
        print(f"Added point to {current_lane}: {x}, {y}")
    
    # Right button click to close the polygon for current lane
    elif event == cv2.EVENT_RBUTTONDOWN and len(lanes[current_lane]) > 2 and not complete[current_lane]:
        # Close the polygon
        complete[current_lane] = True
        print(f"{current_lane} polygon completed!")
        
        # Create and display mask for this lane
        create_mask(current_lane)
    
    # Update the display
    image = temp_img
    cv2.imshow(window_name, image)

def create_mask(lane_name):
    global original_image, lanes
    
    # Create a mask
    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    
    # Convert points list to numpy array
    points_array = np.array([lanes[lane_name]], dtype=np.int32)
    
    # Fill the polygon on the mask
    cv2.fillPoly(mask, points_array, 255)
    
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    
    # Show the mask and masked image
    cv2.imshow(f'Mask - {lane_name}', mask)
    cv2.imshow(f'Masked Image - {lane_name}', masked_image)

def save_to_json(filename='road_lanes_roi.json'):
    global lanes
    
    # Convert tuple points to list for JSON serialization
    lanes_dict = {}
    for lane_name, points in lanes.items():
        if len(points) > 2:  # Only save lanes with at least 3 points
            lanes_dict[lane_name] = [list(point) for point in points]
    
    # Create a dictionary to store
    data = {
        'lanes': lanes_dict
    }
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Lane ROIs saved to {filename}")

def load_from_json(filename='road_lanes_roi.json'):
    global lanes, complete
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to tuples
        loaded_lanes = data.get('lanes', {})
        
        # Reset existing lanes
        lanes = {"lane1": [], "lane2": [], "lane3": []}
        complete = {"lane1": False, "lane2": False, "lane3": False}
        
        # Load points for each lane
        for lane_name, points in loaded_lanes.items():
            if lane_name in lanes:
                lanes[lane_name] = [tuple(point) for point in points]
                if len(lanes[lane_name]) > 2:
                    complete[lane_name] = True
                    create_mask(lane_name)
        
        print(f"Loaded lanes from {filename}")
        return True
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return False

def visualize_all_lanes():
    global original_image, lanes
    
    # Create a copy of the original image
    result = original_image.copy()
    
    # Blue color scheme for lanes
    lane_colors = {
        "lane1": (255, 180, 0),    # Light blue (BGR format)
        "lane2": (255, 100, 0),    # Medium blue
        "lane3": (255, 0, 0)       # Dark blue
    }
    
    # Combined mask for all lanes
    combined_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    
    # Process each lane
    for lane_name, points in lanes.items():
        if len(points) > 2 and complete[lane_name]:
            # Create individual lane mask
            lane_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            points_array = np.array([points], dtype=np.int32)
            cv2.fillPoly(lane_mask, points_array, 255)
            
            # Add to combined mask
            combined_mask = cv2.bitwise_or(combined_mask, lane_mask)
            
            # Color overlay for this lane
            colored_lane = np.zeros_like(original_image)
            colored_lane[:] = lane_colors[lane_name]
            
            # Apply overlay with transparency
            lane_overlay = cv2.bitwise_and(colored_lane, colored_lane, mask=lane_mask)
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(lane_overlay, alpha, result, 1 - alpha, 0, result, mask=lane_mask)
            
            # Draw the lane outline
            cv2.polylines(result, [points_array], True, lane_colors[lane_name], 2)
    
    # Show the final visualization
    cv2.imshow('All Lanes Visualization', result)
    
    # Create masked original image (all lanes)
    masked_original = cv2.bitwise_and(original_image, original_image, mask=combined_mask)
    cv2.imshow('Combined Masked Image', masked_original)

def main():
    global image, original_image, current_lane, lanes, complete
    
    # Load a road image
    image_path = 'image.jpeg'  # Change this to your road image path
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not read the image at {image_path}")
        return
    
    # Make a copy for drawing
    image = original_image.copy()
    
    # Create a window and set the mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Instructions:")
    print("- Left-click to add points to the current lane polygon")
    print("- Right-click to complete the current lane polygon")
    print("- Press '1', '2', or '3' to switch between lanes")
    print("- Press 'v' to visualize all lanes together")
    print("- Press 's' to save all lane points to JSON")
    print("- Press 'l' to load lanes from JSON")
    print("- Press 'r' to reset current lane")
    print("- Press 'a' to reset all lanes")
    print("- Press 'ESC' to exit")
    
    while True:
        # Display the image with current points
        cv2.imshow(window_name, image)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            break
        elif key == ord('1'):  # Switch to lane 1
            current_lane = "lane1"
            print("Switched to Lane 1")
        elif key == ord('2'):  # Switch to lane 2
            current_lane = "lane2"
            print("Switched to Lane 2")
        elif key == ord('3'):  # Switch to lane 3
            current_lane = "lane3"
            print("Switched to Lane 3")
        elif key == ord('r'):  # Reset current lane
            lanes[current_lane] = []
            complete[current_lane] = False
            cv2.destroyWindow(f'Mask - {current_lane}')
            cv2.destroyWindow(f'Masked Image - {current_lane}')
            print(f"Reset {current_lane}")
        elif key == ord('a'):  # Reset all lanes
            lanes = {"lane1": [], "lane2": [], "lane3": []}
            complete = {"lane1": False, "lane2": False, "lane3": False}
            for lane in ["lane1", "lane2", "lane3"]:
                try:
                    cv2.destroyWindow(f'Mask - {lane}')
                    cv2.destroyWindow(f'Masked Image - {lane}')
                except:
                    pass
            print("Reset all lanes")
        elif key == ord('s'):  # Save to JSON
            save_to_json()
        elif key == ord('l'):  # Load from JSON
            load_from_json()
        elif key == ord('v'):  # Visualize all lanes
            visualize_all_lanes()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()