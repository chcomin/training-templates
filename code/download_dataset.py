'''Download and extract a dataset. Can be used as 

python download_dataset.py name_of_dataset
'''

import argparse
from torchvision.datasets.utils import download_and_extract_archive

def vessel_mini(root_folder):

    url = 'https://www.dropbox.com/scl/fi/0nsoze6y24ap893q8l4at/vessel_mini.tar.gz?rlkey=t9hxh7zzh0pg9kfxl58bfer70&dl=1'
    download_root = root_folder
    extract_root = root_folder
    filename = 'vessel_mini.tar.gz'
    download_and_extract_archive(url, download_root, extract_root, filename, remove_finished=True)

def get_args_parser(add_help=True):
    
    parser = argparse.ArgumentParser(description="Download some datasets", add_help=add_help)

    parser.add_argument("dataset", choices=['vessel_mini'], type=str, help="Name of the dataset")
    parser.add_argument("-root", default="../data", type=str, help="Where to put the data")
 
    return parser

def main(args):
    if args.dataset=='vessel_mini':
        vessel_mini(args.root)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
