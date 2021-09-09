import os
import json

def get_img(qa_path, write_path):
    img_name_list = []
    if 'train' in qa_path:
        folder = 'train/'
    else:
        folder = 'val/'
    with open(qa_path, 'r') as f:
        qas = json.load(f)
        for item in qas:
            img_name_list.append('/scratch/mluo26/yankai/output/' + folder + item['img_id'] + '.npy')

    with open(write_path, 'w') as f:
        json.dump(img_name_list, f)

if __name__ == "__main__":
    get_img('train.json', 'train_img_list.json')
    get_img('testdev.json', 'testdev_img_list.json')
