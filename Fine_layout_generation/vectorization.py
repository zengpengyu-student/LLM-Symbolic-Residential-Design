import os

import cv2
import svgwrite
from svgwrite import cm, mm
import numpy as np
import re
from collections import Counter
import json
from natsort import natsorted
from tools.vector_change import split_polyline_by_inflection_rule,split_adherent_doors_with_gap

red_line_low_BGR = np.array([0,0,245])
red_line_up_BGR = np.array([20,20,255])
green_line_low_BGR = np.array([0,100,0])
green_line_up_BGR = np.array([20,150,20])
blue_line_low_BGR = np.array([245,0,0])
blue_line_up_BGR = np.array([255,20,20])
livingroom_low_BGR = np.array([219,231,234])
livingroom_up_BGR = np.array([239,251,254])
bedroom_low_BGR = np.array([160,234,243])
bedroom_up_BGR = np.array([180,254,255])
balcony_low_BGR = np.array([124,205,197])
balcony_up_BGR = np.array([144,225,217])
kitchen_low_BGR = np.array([203,205,224])
kitchen_up_BGR = np.array([223,225,244])
toilet_low_BGR = np.array([241,222,194])
toilet_up_BGR = np.array([255,242,214])
'''
wall_line_BGR = np.array([255,0,0])
windows_line_BGR = np.array([0,133,0])
doors_line_BGR = np.array([255, 0, 0])
livingroom_BGR = np.array([229, 242, 244])
bedroom_BGR = np.array([171, 244, 253])
balcony_BGR = np.array([135, 216, 208])
kitchen_BGR = np.array([214, 216, 234])
toilet_BGR = np.array([252, 233, 205])
'''


def image_to_svg(image, svg_path, type):
    path_data_all = []
    # Read the image and convert it to grayscale
    #image = cv2.imread(image_path)

    # Threshold the image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours and draw their outlines
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create the SVG canvas
    dwg = svgwrite.Drawing(svg_path, profile='tiny')
    kernel = np.ones((3, 3), np.uint8)  # 1-pixel dilation kernel (3x3)
    h, w = gray.shape

    # Iterate over each contour and add it to the SVG
    num = 0
    if type == 'door' or type == 'window':
        for contour in contours:
            # Convert the contour to a polygon path
            path_data = "M"
            for point in contour:
                path_data += f"{point[0][0]},{point[0][1]} "
            path_data += "Z"  # Close the path
            # Add the path to the SVG
            # dwg.add(dwg.path(d=path_data, fill='black'))
            path_data_all.append(path_data)
            num += 1
    else:
        for contour in contours:
            # 1. Create a separate blank mask for the current contour (to avoid affecting other contours)
            single_contour_mask = np.zeros((h, w), dtype=np.uint8)
            # 2. Draw the current contour on its mask (white fill)
            cv2.drawContours(single_contour_mask, [contour], 0, 255, thickness=cv2.FILLED)

            # 3. Dilate only the current contour's mask by 1 pixel

            single_contour_mask_expanded = cv2.dilate(single_contour_mask, kernel, iterations=1)
            # 4. Re-extract the contour from the dilated mask (current contour only)
            expanded_contours, _ = cv2.findContours(single_contour_mask_expanded, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
            # Take the first contour (dilation still leaves a single contour)
            if len(expanded_contours) > 0:
                contour = expanded_contours[0]

            # Convert the contour to a polygon path
            path_data = "M"
            for point in contour:
                path_data += f"{point[0][0]},{point[0][1]} "
            path_data += "Z"  # Close the path

            # Add the path to the SVG
            #dwg.add(dwg.path(d=path_data, fill='black'))
            path_data_all.append(path_data)
            num += 1

    # Save the SVG file
    #dwg.save()
    return path_data_all, num

def extract_block(img, low, up):
    image = cv2.imread(img)
    room = cv2.inRange(image, low, up)
    mask_red = cv2.cvtColor(room, cv2.COLOR_GRAY2BGR)
    room = cv2.bitwise_and(image, mask_red)
    return room

# --------------------Adjust vector paths-------------------------
def extract_coordinates(path):
    """Extract coordinate pairs from an SVG path string."""
    return re.findall(r'\d+,\d+', path)

def remove_corners(path, threshold=1):
    """Remove chamfers from an SVG path while keeping its segments straight."""
    contin = False
    x_0, y_0 = 64, 64
    # Parse coordinate points
    coordinates = re.findall(r'\d+,\d+', path)
    coords = [tuple(map(int, coord.split(','))) for coord in coordinates]

    # Detect and remove chamfers
    i = 0
    while i < len(coords) - 1:
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]

        # Check whether a chamfer is formed
        if abs(x0 - x1) == threshold and abs(y0 - y1) == threshold:
            x_1, y_1 = coords[i - 1]
            if i == len(coords) - 2:
                x2, y2 = coords[0]
            else:
                x2, y2 = coords[i + 2]
            if (x0==x_1 and x1==x2) or (y0==y_1 and y1==y2):
                if x0==x_1 and x1==x2:
                    y_0 = y1
                    x_0 = x0
                elif y0==y_1 and y1==y2:
                    x_0 = x1
                    y_0 = y0
                coords[i] = (x_0, y_0)
            else:
                x_0, y_0 = 0, 0
                # Modify the first coordinate point and remove the second
                x_ij1, y_ij1 = coords[i - 1]
                if i + 2 > len(coords)-1:
                    x_ij2, y_ij2 = coords[0]
                else:
                    x_ij2, y_ij2 = coords[i + 2]
                if abs(x0 - x_ij1) < abs(y0 - y_ij1):
                    x_0 = x0
                else:
                    y_0 = y0
                if abs(x0 - x_ij2) < abs(y0 - y_ij2):
                    x_0 = x1
                else:
                    y_0 = y1
                coords[i] = (x_0, y_0)
                del coords[i + 1]
        else:
            i += 1

    # Rebuild the path
    new_path = ' '.join([f'{x},{y}' for x, y in coords])
    if x_0==0 or y_0==0:
        contin = True
    return f'M{new_path} Z', contin

def count_coordinates(coordinates):
    """Count the occurrences of each X and Y coordinate."""
    x_counts = Counter()
    y_counts = Counter()
    for coord in coordinates:
        x, y = map(int, coord.split(','))
        x_counts[x] += 1
        y_counts[y] += 1
    return x_counts, y_counts

def find_repeated_coordinates(x_counts, y_counts, count=2):
    """Find X and Y coordinates that occur a specified number of times."""
    x_repeated = [x for x, c in x_counts.items() if c == count]
    y_repeated = [y for y, c in y_counts.items() if c == count]
    return x_repeated, y_repeated

def parse_path(path):
    """Parse an SVG path string and extract coordinate points."""
    coordinates = re.findall(r'\d+,\d+', path)
    return [tuple(map(int, coord.split(','))) for coord in coordinates]

def adjust_path(path, x_repeated, y_repeated, threshold=2):
    """Adjust the path based on reference coordinates."""
    coords = parse_path(path)
    adjusted_coords = []
    for x, y in coords:
        new_x = x
        new_y = y
        for xr in x_repeated:
            if abs(x - xr) <= threshold:
                new_x = xr
        for yr in y_repeated:
            if abs(y - yr) <= threshold:
                new_y = yr
        adjusted_coords.append((new_x, new_y))
    adjusted_path = ' '.join([f'{x*4},{y*4}' for x, y in adjusted_coords])
    return f'M{adjusted_path} Z'

def extract_block_later(path_data_room, x_all, y_all, type):
    room_extract = []
    contin = False
    for path in path_data_room:
        # Adjust the bedroom path
        path, contin = remove_corners(path)
        if contin:
            break
        if type in ('door', 'window'):
            adjusted_bedroom_path = adjust_path(path, x_all, y_all, threshold=0)
        else:
            adjusted_bedroom_path = adjust_path(path, x_all, y_all, threshold=1)
        coordinates = extract_coordinates(adjusted_bedroom_path)
        x_counts, y_counts = count_coordinates(coordinates)
        x_repeated, y_repeated = find_repeated_coordinates(x_counts, y_counts)
        x_all.extend(x_repeated)
        y_all.extend(y_repeated)
        room_extract.append(adjusted_bedroom_path)
    return room_extract, x_all, y_all, contin


path_base = f'../test/Fine_output/cache_img/'
json_path = f'../test/Fine_output/cache_json/'
if not os.path.exists(json_path):
    os.mkdir(json_path)
img_list = natsorted(os.listdir(path_base))
True_all = 0
False_all = 0
for n, img_name in enumerate(img_list):
    index = img_name.split('.')[0]
    with open(f'{json_path}/{index}.json', 'w') as f:
        #path_image_all = path_base+img_name
        path_image_all = path_base+f'{img_name}'

        living_room = extract_block(path_image_all, livingroom_low_BGR, livingroom_up_BGR)
        bedroom = extract_block(path_image_all, bedroom_low_BGR, bedroom_up_BGR)
        balcony = extract_block(path_image_all, balcony_low_BGR, balcony_up_BGR)
        kitchen = extract_block(path_image_all, kitchen_low_BGR, kitchen_up_BGR)
        toilet = extract_block(path_image_all, toilet_low_BGR, toilet_up_BGR)
        door = extract_block(path_image_all, blue_line_low_BGR, blue_line_up_BGR)
        window = extract_block(path_image_all, green_line_low_BGR, green_line_up_BGR)
        save_path = 'output.svg'
        # Use the function
        path_data_bedroom, bedroom_num  = image_to_svg(bedroom, save_path, 'bedroom')
        path_data_balcony, balcony_num = image_to_svg(balcony, save_path, 'balcony')
        path_data_kitchen, kitchen_num = image_to_svg(kitchen, save_path, 'kitchen')
        path_data_toilet, toilet_num = image_to_svg(toilet, save_path, 'toilet')
        path_data_living_room, living_room_num = image_to_svg(living_room, save_path, 'living_room')
        path_data_door, door_num = image_to_svg(door, save_path, 'door')
        path_data_window, window_num = image_to_svg(window, save_path, 'window')
        '''
        print('bedroom:', path_data_bedroom)
        print('balcony:', path_data_balcony)
        print('kitchen:', path_data_kitchen)
        print('toilet:', path_data_toilet)
        print('living_room:', path_data_living_room)
        print('door:', path_data_door)
        print('window:', path_data_window)
        '''

        # Example path
        paths = {
            'living_room': path_data_living_room,
        }

        # Adjust the path
        # Extract and count coordinates for each path
        x_all = []
        y_all = []

        '''
        living_room_extract = []
        for path in path_data_living_room:
            path, contin_living_room = remove_corners(path)
            coordinates = extract_coordinates(path)
            x_counts, y_counts = count_coordinates(coordinates)
            x_repeated, y_repeated = find_repeated_coordinates(x_counts, y_counts)
            x_all.extend(x_repeated)
            y_all.extend(y_repeated)
            living_room_extract.append(path)
        '''

        door_extract, x_all, y_all, contin_door = extract_block_later(path_data_door, x_all, y_all, 'door')
        window_extract, x_all, y_all, contin_window = extract_block_later(path_data_window, x_all, y_all, 'window')
        living_room_extract, x_all, y_all, contin_living_room = extract_block_later(path_data_living_room, x_all, y_all, 'living_room')
        bedroom_extract, x_all, y_all, contin_bedroom = extract_block_later(path_data_bedroom, x_all, y_all, 'bedroom')
        kitchen_extract, x_all, y_all, contin_kitchen = extract_block_later(path_data_kitchen, x_all, y_all, 'kitchen')
        toilet_extract, x_all, y_all, contin_toilet = extract_block_later(path_data_toilet, x_all, y_all, 'toilet')
        balcony_extract, x_all, y_all, contin_balcony = extract_block_later(path_data_balcony, x_all, y_all, 'balcony')
        '''
        print('living_room_extract:', living_room_extract)
        print('bedroom_extract:', bedroom_extract)
        print('kitchen_extract:', kitchen_extract)
        print('toilet_extract:', toilet_extract)
        print('balcony_extract:', balcony_extract)
        print('door_extract:', door_extract)
        print('window_extract:', window_extract)
        '''
        # The earlier condition can be omitted; output everything instead.
        #if contin_living_room or contin_bedroom or contin_kitchen or contin_toilet or contin_balcony or contin_door or contin_window:
        #    print(f'Error!, n:{n}')
        #    False_all += 1
        #    continue
        #True_all += 1
        #data = {"Pic": str(f'{img_name}'),
        #        "Prompt": str(f'Living_room: {living_room_extract}; Bedroom: {bedroom_extract}; Kitchen: {kitchen_extract}; Toilet: {toilet_extract}; Balcony: {balcony_extract}; Door: {door_extract}; Window: {window_extract}'),
        #        }
        #prompt = str(f'Living_room: {living_room_extract}; Bedroom: {bedroom_extract}; Kitchen: {kitchen_extract}; Toilet: {toilet_extract}; Balcony: {balcony_extract}; Door: {door_extract}; Window: {window_extract}')
        #data = {f"{img_name}": [prompt]}
        data = {"Pic": str(f'{img_name}'),
                "living_room": living_room_extract,
                "bedroom": bedroom_extract,
                "kitchen": kitchen_extract,
                "bathroom": toilet_extract,
                "balcony": balcony_extract,
                "door": door_extract,
                "window": window_extract,
                }
        data = split_adherent_doors_with_gap(data)
        data = split_polyline_by_inflection_rule(data)
        '''
        formatted_output = "{\n"
        formatted_output += f'    "Pic": "{img_name}",\n'
        formatted_output += f'    "living_room": "{living_room_extract}",\n'
        formatted_output += f'    "bedroom": "{bedroom_extract}",\n'
        formatted_output += f'    "kitchen": "{kitchen_extract}",\n'
        formatted_output += f'    "toilet": "{toilet_extract}",\n'
        formatted_output += f'    "balcony": "{balcony_extract}",\n'
        formatted_output += f'    "door": "{door_extract}",\n'
        formatted_output += f'    "window": "{window_extract}"\n'
        formatted_output += "}"
        #json.dump(data, f)
        f.write(formatted_output)
        '''
        json.dump(data, f)
        #f.write('\n')
        #print(f'n:{n}; True_all:{True_all}; False_all:{False_all}')
        #if n >= 71126:
        #    break

