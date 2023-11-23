import os
import json
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]

def process_files(input_folder, output_file, sorted_file):
    with open(output_file, 'w') as f_out:
        for folder_name in os.listdir(input_folder):
            folder_path = os.path.join(input_folder, folder_name)

            if os.path.isdir(folder_path):
                meta_path = os.path.join(folder_path, 'meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as meta_file:
                        meta_data = json.load(meta_file)
                        model_cat = meta_data.get("model_cat", "N/A")

                    f_out.write(f"{folder_name} {model_cat}\n")

    with open(output_file, 'r') as f:
        lines = f.readlines()
        sorted_lines = sorted(lines)

    with open(sorted_file, 'w') as f:
        f.writelines(sorted_lines)


if __name__=='__main__':
    input_floder = os.path.join(rootPath, 'data', 'partnet-mobility-v0', 'dataset')
    output_file = os.path.join(rootPath, 'stats', 'all_shapeid_cat_map.txt')
    sorted_file = os.path.join(rootPath, 'stats', 'sorted_shapeid_cat_map.txt')
    process_files(input_floder, output_file, sorted_file)