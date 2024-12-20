import json
import argparse

def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data, data_path):
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--mm_image_tower', type=str, default=None)

    args = parser.parse_args()

    config = read_json(args.path+'/config.json')
    config['mm_image_tower'] = args.mm_image_tower
    write_json(config, args.path+'/config.json')
    print(f"mm_image_tower in {args.path+'/config.json'} has been changed to {args.mm_image_tower}.")

    