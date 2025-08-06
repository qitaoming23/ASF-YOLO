from operator import ne
import os
import shutil
import argparse

def check_subfolders_for_file(parent_folder, file_name):
    for root, dirs, files in os.walk(parent_folder):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            file_path = os.path.join(dir_path, file_name)
            if not os.path.exists(file_path):
                print(f"The file '{file_name}' does not exist in the folder '{dir_path}'")
                return False
    return True

def main(source_folder, target_folder):
    # 获取源文件夹下所有包含'image'的文件夹
    image_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f)) and 'image' in f]

    # 逐一移动文件夹
    for folder in image_folders:
        source_path = os.path.join(source_folder, folder)
        target_path = os.path.join(target_folder, 'JPEGImages', folder)
        shutil.move(source_path, target_path)
        print(f'Moved {folder} to {target_path}')

    # 重命名labels文件夹为Annotations
    old_labels_path = os.path.join(source_folder, 'labels')
    new_labels_path = os.path.join(source_folder, 'VocFormat', 'Annotations')
    os.rename(old_labels_path, new_labels_path)
    # 获取指定文件夹下的文件列表
    files = os.listdir(new_labels_path)
    # 初始化变量
    train_val = []
    tv = ''
    test = []
    ts = ''
    train = []
    tr = ''
    val = []
    v = ''
    # 遍历文件列表
    for i in files:
        a = i.split('.')
        # 检查文件是否存在
        if check_subfolders_for_file(target_path, a[0]+'.tif'):
            # 使用哈希函数将文件分配到测试集或训练集
            if hash(a[0]) % 10 >= 1:
                train_val.append(a[0])
                tv += a[0] + '\n'
                if hash(a[0]) % 10 >= 2:
                    train.append(a[0])
                    tr += a[0] + '\n'
                else:
                    val.append(a[0])
                    v += a[0] + '\n'
            else:
                test.append(a[0])
                ts += a[0] + '\n'
        else:
            # 输出文件不存在的路径
            print(os.path.join(target_path, a[0]+'.tif') + " does not exist")
            # 可选：删除不存在的文件
            # os.remove(os.path.join(root_path, 'labels', a[0]+'.xml'))


    output_path = os.path.join(source_folder, 'VocFormat', 'ImageSets/Main/')
    print(output_path)
    if os.path.exists(output_path):
        pass
    else:
        os.makedirs(output_path)

    wf=open(os.path.join(output_path, "train.txt"),'w+',encoding='utf-8')
    wf.write(tr)
    wf.close()

    wf=open(os.path.join(output_path, "trainval.txt"),'w+',encoding='utf-8')
    wf.write(tv)
    wf.close()

    wf=open(os.path.join(output_path, "val.txt"),'w+',encoding='utf-8')
    wf.write(v)
    wf.close()

    wf=open(os.path.join(output_path, "test.txt"),'w+',encoding='utf-8')
    wf.write(ts)
    wf.close()

    print(f'train {len(train)}')
    print(f'val {len(val)}')
    print(f'test{len(test)}')
    print(f'sum {len(train_val)+len(test)}')

if __name__ == '__main__':
    argparses = argparse.ArgumentParser()
    argparses.add_argument('--source_folder', type=str, default='/home/wdblink/Dataset/RGB-DSM')
    argparses.add_argument('--target_folder', type=str, default='/home/wdblink/Dataset/RGB-DSM/VocFormat')
    args = argparses.parse_args()
    source_folder = args.source_folder
    target_folder = args.target_folder

    main(source_folder, target_folder)