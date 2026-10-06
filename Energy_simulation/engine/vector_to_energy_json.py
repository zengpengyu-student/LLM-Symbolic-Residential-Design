import re


def process_coordinate_format(path_str):

    pattern = r'(\d+)'


    def replace_with_float(match):
        return f"{match.group(1)}.0"


    return re.sub(pattern, replace_with_float, path_str)


def convert_floor_plan_format(original_dict):

    key_mapping = [
        ("living_room", "livingroom"),
        ("bedroom", "bedroom"),
        ("kitchen", "kitchen"),
        ("bathroom", "toilet"),
        ("balcony", "balcony")
    ]


    result_dict = {
        "Pic": original_dict["Pic"]
    }


    for original_key, target_prefix in key_mapping:
        if original_key not in original_dict:
            continue
        element_list = original_dict[original_key]

        for idx, element in enumerate(element_list):
            new_key = f"{target_prefix}_{idx}"
            processed_element = process_coordinate_format(element)

            result_dict[new_key] = str([processed_element])


    door_list = [process_coordinate_format(item) for item in original_dict["door"]]
    result_dict["door"] = str(door_list)


    window_list = [process_coordinate_format(item) for item in original_dict["window"]]
    result_dict["window"] = str(window_list)

    return result_dict
