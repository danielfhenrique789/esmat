import os

def get_base_path():
    return os.path.dirname(os.path.abspath(__file__)) 

def get_folders_from_path(_path):
    list_folders = []
    if(os.path.isdir(_path)):
        for dir in os.listdir(_path):
            if(os.path.isdir(os.path.join(_path, dir))):
                list_folders.append(dir)
    return list_folders

def file_exists(_file):
    return os.path.exists(_file)