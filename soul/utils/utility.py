import os
import datetime

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

# TODO draw result analysis pictures
