"""Fire-safety distance analysis for raster floor plans."""

import os

import cv2
import numpy as np
import heapq
import math

def auto_calculate_scale(image, real_wall_thickness_mm=150, real_door_width_mm=900):
              
    lower_red = np.array([0, 0, 128])
    upper_red = np.array([100, 100, 255])
    red_mask = cv2.inRange(image, lower_red, upper_red)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(red_mask, kernel, iterations=1)
    eroded = cv2.erode(red_mask, kernel, iterations=1)
    wall_thickness = cv2.subtract(dilated, eroded)

    nonzero_coords = np.column_stack(np.where(wall_thickness > 0))
    if len(nonzero_coords) == 0:
        avg_wall_thickness_px = None
        print("No red wall boundary was detected.")
    else:
        avg_wall_thickness_px = np.mean([
            cv2.countNonZero(wall_thickness[max(0, y - 1):y + 2, max(0, x - 1):x + 2])
            for y, x in nonzero_coords
        ])
        print(f"Detected wall thickness: {avg_wall_thickness_px:.2f} px")

            
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 100, 100])
    blue_mask = cv2.inRange(image, lower_blue, upper_blue)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    door_widths_px = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        door_width = max(w, h)
        door_widths_px.append(door_width)

    if door_widths_px:
        avg_door_width_px = np.mean(door_widths_px)
        print(f"Detected door width: {avg_door_width_px:.2f} px")
    else:
        avg_door_width_px = None
        print("No blue door line was detected.")

              
    pixel_lengths = []
    real_lengths_mm = []
    if avg_wall_thickness_px:
        pixel_lengths.append(avg_wall_thickness_px)
        real_lengths_mm.append(real_wall_thickness_mm)
    if avg_door_width_px:
        pixel_lengths.append(avg_door_width_px)
        real_lengths_mm.append(real_door_width_mm)

    if len(pixel_lengths) == 0:
        print("No valid building components were detected; using the default scale of 30 mm per pixel.")
        return 30.0

    scale_factors = [real / pix for pix, real in zip(pixel_lengths, real_lengths_mm)]
    scale_mm_per_pixel = sum(scale_factors) / len(scale_factors)
    print(f"Fitted scale: 1 px ≈ {scale_mm_per_pixel:.2f} mm")
    return scale_mm_per_pixel

     
def upscale_image(image, scale=2):
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

                                       
def extract_building_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

             
    kernel = np.ones((3, 3), np.uint8)
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

          
    contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

               
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)           

    return contours, contour_image


                                      
def detect_entrance(image, contours):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([130, 255, 255])
    deep_blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

           
    kernel = np.ones((3, 3), np.uint8)
    deep_blue_mask = cv2.morphologyEx(deep_blue_mask, cv2.MORPH_OPEN, kernel)

            
    deep_blue_lines = cv2.bitwise_and(image, image, mask=deep_blue_mask)

             
    deep_blue_gray = cv2.cvtColor(deep_blue_lines, cv2.COLOR_BGR2GRAY)
    _, deep_blue_thresh = cv2.threshold(deep_blue_gray, 1, 255, cv2.THRESH_BINARY)
    deep_blue_contours, _ = cv2.findContours(deep_blue_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

           
    door_marked_image = image.copy()
    entrance_position = None

                
    for blue_contour in deep_blue_contours:
                    
        for contour in contours:
            distances = [
                cv2.pointPolygonTest(
                    contour,
                    (int(point[0][0]), int(point[0][1])),
                    True
                )
                for point in blue_contour
            ]
                                 
            if any(-2 <= d <= 2 for d in distances):
                               
                cv2.drawContours(door_marked_image, [blue_contour], -1, (255, 255, 255), 3)
                entrance_position = tuple(blue_contour[0][0])
                                        
                M = cv2.moments(blue_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    entrance_position = (cx, cy)
                else:
                                                   
                                       
                    entrance_position = tuple(blue_contour[0][0])

                                        
                break
    return entrance_position, deep_blue_lines, door_marked_image

                                        
def generate_walkable_grid(image, entrance_position):
            
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                           
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

                
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                         
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

                         
    kernel = np.ones((3, 3), np.uint8)
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)
    mask_blue = cv2.erode(mask_blue, kernel, iterations=1)

                     
    binary[mask_blue > 0] = 0

                      
    walkable_grid_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return binary, walkable_grid_colored

                            
def compute_distance_map(occupancy, entrance_position):
    """Compute 8-connected shortest paths with Euclidean grid-step costs."""
    h, w = occupancy.shape
    distances = np.full((h, w), np.inf, dtype=np.float64)
    start = (entrance_position[1], entrance_position[0])              
    distances[start] = 0
    queue = [(0.0, start[0], start[1])]
                   
    directions = [(1,0), (-1,0), (0,1), (0,-1),
                  (1,1), (1,-1), (-1,1), (-1,-1)]
    while queue:
        current_distance, row, col = heapq.heappop(queue)
        if current_distance > distances[row, col]:
            continue

        for d in directions:
            next_row = row + d[0]
            next_col = col + d[1]
            if 0 <= next_row < h and 0 <= next_col < w:
                if occupancy[next_row, next_col] == 0:
                    step_cost = math.sqrt(2) if d[0] != 0 and d[1] != 0 else 1.0
                    next_distance = current_distance + step_cost
                    if next_distance < distances[next_row, next_col]:
                        distances[next_row, next_col] = next_distance
                        heapq.heappush(queue, (next_distance, next_row, next_col))

    distances[~np.isfinite(distances)] = -1.0
    return distances

                     
def a_star(occupancy, start, goal):
    h, w = occupancy.shape

    def move_cost(current, neighbor):
        if current[0] != neighbor[0] and current[1] != neighbor[1]:
            return math.sqrt(2)
        else:
            return 1

    def heuristic(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    directions = [(1,0), (-1,0), (0,1), (0,-1),
                  (1,1), (1,-1), (-1,1), (-1,-1)]               

    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w:
                if occupancy[neighbor] == 0:           
                    tentative_g = current_g + move_cost(current, neighbor)
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                        came_from[neighbor] = current
    return None

def draw_firetest(image, image_path, entrance_position, farthest_point_xy, occupancy, max_distance, walkable_grid):
                             
    start_rc = (entrance_position[1], entrance_position[0])
    goal_rc = (farthest_point_xy[1], farthest_point_xy[0])
    path_astar = a_star(occupancy, start_rc, goal_rc)

    if path_astar is None:
        print("No path was found from the entrance to the farthest point.")
    else:
                          
        path_xy = [(pt[1], pt[0]) for pt in path_astar]

                               
        total_cost = 0
        for i in range(1, len(path_astar)):
            dr = abs(path_astar[i][0] - path_astar[i - 1][0])
            dc = abs(path_astar[i][1] - path_astar[i - 1][1])
            if dr == 1 and dc == 1:
                step_cost = math.sqrt(2)
            else:
                step_cost = 1
            total_cost += step_cost

                       
        scale_mm_per_pixel = auto_calculate_scale(image)
        real_distance_mm = max_distance / 3.0
        pic_m = 12           

                    
        if real_distance_mm < (pic_m * 20):
            safety_msg = "Fire protection meets the standard <20m"
            safety_color = (0, 255, 0)
        else:
            safety_msg = "Fire protection does not meet the standards"
            safety_color = (0, 0, 255)

                                                       
        marked_grid_with_astar = walkable_grid.copy()
        pts = np.array(path_xy, np.int32).reshape((-1, 1, 2))
        cv2.polylines(marked_grid_with_astar, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

                      
                    
                                                                                                                                                                       
                                     
                                                                                                                  

        img_name = os.path.basename(image_path)
        cv2.imwrite(f'../stage5_output/firetest/{img_name}', marked_grid_with_astar)
                                   
                                                                             
                                                                                 
                         
                    

def max_distance_class(image, image_path):
    image = upscale_image(image, scale=1)

           
                                          
                                                        
                                  
                

                                           
    contours, contour_image = extract_building_contour(image)

                           
    gray_for_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    building_mask = np.zeros_like(gray_for_mask, dtype=np.uint8)
    cv2.drawContours(building_mask, contours, -1, 255, thickness=-1)          

               
                                                  
                                
                                                                
                                         
                     
                

                                          
    entrance_position, deep_blue_lines, door_marked_image = detect_entrance(image, contours)

                 
                                                  
                                
                                                                    
                                          
                     
                

                                            
    binary, walkable_grid = generate_walkable_grid(image, entrance_position)

                
    if entrance_position is not None:
        ex, ey = entrance_position
        cv2.circle(walkable_grid, (ex, ey), 5, (0, 0, 255), -1)               

             
                                                  
                                
                                                                
                                                              
                     
                

                                                                                      
    occupancy = np.ones_like(binary, dtype=np.uint8)              
    inside = (binary == 0) & (building_mask == 255)
    occupancy[inside] = 0          

    distances = compute_distance_map(occupancy, entrance_position)
    max_distance = np.max(distances)
    farthest_idx = np.argmax(distances)
    farthest_point = np.unravel_index(farthest_idx, distances.shape)              
    farthest_point_xy = (farthest_point[1], farthest_point[0])              

    draw_firetest(image, image_path, entrance_position, farthest_point_xy, occupancy, max_distance, walkable_grid)

                                             
                                             

                                  
                                      
    return max_distance
