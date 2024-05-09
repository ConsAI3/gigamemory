# from https://github.com/beir-cellar/beir/blob/main/beir/util.py
from tqdm.autonotebook import tqdm
import logging
import os
import requests
import zipfile
import tarfile
import bz2
from typing import Optional

logger = logging.getLogger(__name__)


def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get('Content-Length', 0))
    with open(save_path, 'wb') as fd, tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True,    
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)

def extract_archive(archive_file: str, subfolder_name: Optional[str], out_dir: str):
    _, ext = os.path.splitext(archive_file)
    if ext == ".zip":
        with zipfile.ZipFile(archive_file, "r") as zip_:
            if subfolder_name is None:
                zip_.extractall(path=out_dir)
            else:
                for member in zip_.namelist():
                    if member.startswith(subfolder_name + "/"):
                        zip_.extract(member, path=out_dir)
    elif ext == ".bz2":
        with tarfile.open(archive_file, "r:bz2") as tar:
            if subfolder_name is None:
                tar.extractall(path=out_dir)
            else:
                total = 0
                for member in tar:
                    total += 1
                    
                print(total)
                # for member in tar:
                #     print(member.name)
                #     # if subfolder_name in member.name:
                #     #     print(member.name)
                #     if member.name.endswith('.bz2'):
                #         tar.extract(member, path=out_dir)
                #         filepath = os.path.join(out_dir, member.name)
                #         newfilepath = filepath[:-4] + ".json"  # remove .bz2 extension
                #         with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
                #             for data in iter(lambda : file.read(100 * 1024), b''):
                #                 new_file.write(data)
                #         os.remove(filepath)  # remove the .bz2 file
                       
                       
                  
    else:
        raise ValueError(f"Unsupported archive format: {ext}")
def download_and_extract(url: str, subfolder_name: Optional[str], out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)
    
    if not os.path.isfile(zip_file):
        logger.info("Downloading {} ...".format(dataset))
        download_url(url, zip_file, chunk_size)
    
    if not os.path.isdir(zip_file.replace(".zip", "")):
        logger.info("Unzipping {} ...".format(dataset))
        extract_archive(zip_file, subfolder_name, out_dir)
    
    return os.path.join(out_dir, dataset.replace(".zip", ""))

