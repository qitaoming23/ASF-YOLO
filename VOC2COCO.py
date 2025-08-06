import shutil
import xml.etree.ElementTree as ET
import os
import sys
from PIL.Image import ImagePointHandler
from tqdm import tqdm
from utils.general import download, Path
import argparse

class_names = ['1']

def convert_label(path, lb_path, year, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(f'{path}/Annotations/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in class_names:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = class_names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

def main(path):
    # Convert
    for spectial, image_set in (('images', 'train'), ('images', 'val'), ('images', 'trainval'), ('images', 'test'),
                            ('images2', 'train'), ('images2', 'val'), ('images2', 'trainval'), ('images2', 'test')):
        # imgs_path = dir / 'images' / f'{image_set}{year}'
        imgs_path = os.path.join(path, 'images', f'{image_set}_{spectial}')
        # lbs_path = dir / 'labels' / f'{image_set}{year}'
        lbs_path = os.path.join(path, 'labels', f'{image_set}_{spectial}')
        if os.path.exists(imgs_path):
            pass
        else:
            os.makedirs(imgs_path)
            os.makedirs(lbs_path)
        # imgs_path.mkdir(exist_ok=True, parents=True)
        # lbs_path.mkdir(exist_ok=True, parents=True)
        image_ids = open(f'{path}/ImageSets/Main/{image_set}.txt').read().strip().split()
        for id in tqdm(image_ids, desc=f'{image_set}{spectial}'):
            f = f'{path}/JPEGImages/{spectial}/{id}.tif'  # old img path
            # lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
            lb_path = os.path.join(lbs_path, f"{id}.txt")
            shutil.copyfile(f, f'{imgs_path}/{id}.tif')
            convert_label(path, lb_path, spectial, id)  # convert labels to YOLO format


if __name__ == '__main__':
    argparses = argparse.ArgumentParser()
    argparses.add_argument('--source_folder', type=str, default='/home/wdblink/Dataset/RGB-DSM/VocFormat')
    args = argparses.parse_args()
    source_folder = args.source_folder
    main(source_folder)