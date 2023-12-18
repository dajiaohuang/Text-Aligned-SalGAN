#this code is modified from 'https://github.com/gumusserv/CLIP-SalGan'
import json
import os

def get_Data(image_directory_path, target_directory_path):

    image_paths = [image_directory_path + "/" + f for f in os.listdir(image_directory_path) if os.path.isfile(os.path.join(image_directory_path, f))]

    image_paths = image_paths[0 : ]
    target_paths = [target_directory_path + "/" + f for f in os.listdir(target_directory_path) if os.path.isfile(os.path.join(target_directory_path, f))]

    target_paths = target_paths[0 : ]
    with open('text.json', 'r') as f:
        text_dic = json.load(f)

    text_descriptions = []

    for path in target_paths:
        path = path[path.rfind('/') + 1:]
        first_nonzero_index = None

        for i in range(len(path)):
            if path[i] != '0':
                first_nonzero_index = i
                break
        if first_nonzero_index != None:
            path = path[first_nonzero_index:]
        
        path = path[:path.find('.') ]
        
        text_descriptions.append(text_dic[path])
    return image_paths, target_paths, text_descriptions