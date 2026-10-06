import pandas as pd

def parse_eso_file(eso_path):

    variables_dict = {}
    zones = set()


    time_data = []
    value_data = []


    with open(eso_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_data_section = False
    current_time_info = None
    data_entries = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("End of Data Dictionary"):
            in_data_section = True
            continue

        if not in_data_section:

            if ',' in line and not line.startswith('Program'):
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        var_id = int(parts[0].strip())

                        var_desc = ','.join(parts[2:]) if len(parts) > 2 else ''


                        room_name = var_desc.split(',')[0].strip() if var_desc else ""


                        if "Zone Ideal Loads Zone Total Heating Energy" in var_desc:

                            base_room = room_name.replace('IDEAL_', '')
                            zones.add(base_room)
                            variables_dict[var_id] = {
                                'type': 'heating',
                                'room': base_room,
                                'description': var_desc
                            }

                        elif "Zone Ideal Loads Zone Total Cooling Energy" in var_desc:

                            base_room = room_name.replace('IDEAL_', '')
                            zones.add(base_room)
                            variables_dict[var_id] = {
                                'type': 'cooling',
                                'room': base_room,
                                'description': var_desc
                            }

                        elif "Zone Lights Electricity Energy" in var_desc:

                            base_room = room_name
                            zones.add(base_room)
                            variables_dict[var_id] = {
                                'type': 'lighting',
                                'room': base_room,
                                'description': var_desc
                            }

                    except ValueError:
                        continue
        else:

            parts = [part.strip() for part in line.split(',')]

            if len(parts) >= 2:
                try:
                    var_id = int(parts[0])


                    if var_id == 2 and len(parts) >= 9:

                        if current_time_info and data_entries:
                            time_data.append(current_time_info)


                            time_step_data = {}
                            for entry in data_entries:
                                entry_var_id, value = entry
                                if entry_var_id in variables_dict:
                                    var_info = variables_dict[entry_var_id]
                                    col_name = f"{var_info['room']}_{var_info['type']}"
                                    time_step_data[col_name] = float(value)

                            value_data.append(time_step_data)


                        data_entries = []


                        current_time_info = {
                            'Day of Simulation': int(parts[1]),
                            'Month': int(parts[2]),
                            'Day of Month': int(parts[3]),
                            'DST Indicator': int(parts[4]),
                            'Hour': int(parts[5]),
                            'StartMinute': float(parts[6]),
                            'EndMinute': float(parts[7]),
                            'DayType': parts[8]
                        }

                    elif var_id in variables_dict:

                        data_entries.append((var_id, parts[1]))

                except (ValueError, IndexError, Exception) as e:
                    continue


    if current_time_info and data_entries:
        time_data.append(current_time_info)
        time_step_data = {}
        for entry in data_entries:
            entry_var_id, value = entry
            if entry_var_id in variables_dict:
                var_info = variables_dict[entry_var_id]
                col_name = f"{var_info['room']}_{var_info['type']}"
                time_step_data[col_name] = float(value)
        value_data.append(time_step_data)


    time_df = pd.DataFrame(time_data)
    values_df = pd.DataFrame(value_data)


    if not time_df.empty and not values_df.empty:
        result_df = pd.concat([time_df, values_df], axis=1)
    else:
        result_df = pd.DataFrame()

    return result_df, zones, variables_dict


def organize_data_by_category(df, zones):
    if df.empty:
        return df


    ordered_columns = []


    time_cols = ['Day of Simulation', 'Month', 'Day of Month',
                 'DST Indicator', 'Hour', 'StartMinute', 'EndMinute', 'DayType']
    for col in time_cols:
        if col in df.columns:
            ordered_columns.append(col)


    categories = ['heating', 'cooling', 'lighting']

    for category in categories:

        category_cols = [col for col in df.columns if col.endswith(f'_{category}')]


        category_cols_sorted = []
        for zone in sorted(zones):
            zone_category_cols = [col for col in category_cols if col.startswith(f'{zone}_')]
            category_cols_sorted.extend(sorted(zone_category_cols))

        ordered_columns.extend(category_cols_sorted)


    existing_columns = [col for col in ordered_columns if col in df.columns]
    organized_df = df[existing_columns]

    return organized_df
