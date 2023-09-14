'''Download and extract a dataset. Can be used as 

python download_dataset.py name_of_dataset
'''
from pathlib import Path
import shutil
import os
import argparse
import torchvision
from torchvision.datasets.utils import download_and_extract_archive

def vessel_mini(root_folder):

    url = 'https://www.dropbox.com/scl/fi/0nsoze6y24ap893q8l4at/vessel_mini.tar.gz?rlkey=t9hxh7zzh0pg9kfxl58bfer70&dl=1'
    download_root = root_folder
    extract_root = root_folder
    filename = 'vessel_mini.tar.gz'
    download_and_extract_archive(url, download_root, extract_root, filename, remove_finished=True)

def oxford_pets(root_folder):

    root_ds = root_folder/'oxford-iiit-pet'
    label_in_folder = root_ds/'annotations/trimaps'
    label_out_folder = root_ds/'labels'
    class_info = root_ds/'annotations/list.txt'

    ds_tv = torchvision.datasets.OxfordIIITPet(root_folder, split='trainval', target_types=('category','segmentation'), download=True)
    print('Organizing dataset')
    shutil.copy2(class_info, root_ds)
    os.makedirs(label_out_folder, exist_ok=True)
    for file in os.listdir(label_in_folder):
        if file[:2]!='._':
            shutil.copy2(label_in_folder/file, label_out_folder/file)

    for file in os.listdir(root_ds/'images'):
        if '.mat' in file:
            os.remove(root_ds/f'images/{file}')

    shutil.rmtree(root_ds/'annotations')
    print('Done!')

def get_args_parser(add_help=True):
    
    parser = argparse.ArgumentParser(description="Download some datasets", add_help=add_help)

    parser.add_argument("dataset", choices=['vessel_mini', 'oxford_pets'], type=str, help="Name of the dataset")
    parser.add_argument("-root", default="../data", type=str, help="Where to put the data")
 
    return parser

def main(args):
    if args.dataset=='vessel_mini':
        vessel_mini(args.root)
    if args.dataset=='oxford_pets':
        oxford_pets(args.root)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
