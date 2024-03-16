from torch.utils.data import IterableDataset

import tarfile
from tarfile import TarFile
from PIL import Image
import json
from io import BytesIO

def read_from_tar(tar, path):
    file = tar.extractfile(path)
    content = file.read()
    return BytesIO(content)

def default_pil_decode(content : BytesIO):
    return Image.open(content)

def default_json_decode(content : BytesIO):
    return json.load(content)

def default_txt_decode(content : BytesIO):
    return content.read().decode("utf-8")

ext_decode_map = {
    ".json" : default_json_decode,
    ".jpg" : default_pil_decode,
    ".jpeg" : default_pil_decode,
    ".png" : default_pil_decode,
    ".txt" : default_txt_decode
}

class S3Dataset(IterableDataset):
    """
    :param s3_url: URL to S3 dataset
    :param filter_keys: Iterable of keys within tar files that we are actually interested in
    (keys = file extensions or common suffixes, ala .jpg or _chosen.jpg)
    :param decodes: Iterable of functions that can be applied to files obtained from above keys
    (if this is not passed, we can try to guess from keys (i.e. if they're text, or json or jpg)
    """
    def __init__(self, s3_url, filter_keys, decodes = None):
        s3 = boto3.resource('s3')
        bucket_name, prefix = s3_url.replace("s3://", "").split("/", 1)
        bucket = s3.Bucket(bucket_name)
        self.objects = iter(bucket.objects.filter(Prefix=prefix))

        if decodes is None:
            decodes = []
            for key in filter_keys:
                for ext in ext_decode_map:
                    if key.endswith(ext):
                        decodes.append(ext_decode_map[ext])
                        break

        self.filter_keys = filter_keys
        self.decodes = decodes

        # Cache is dict (over keys) of iterators over current tar file
        # There is a unique iterator for each key
        self.tar_cache = self.get_next_tar()

    def create_tar_iter(self, tarfile : TarFile, key : str):
        for member in tarfile.getmembers():
            if member.isfile() and member.name.endswith(key):
                yield read_from_tar(tarfile, member.name)
    
    def get_next_tar(self):
        try:
            while True:
                obj = next(self.objects)
                if obj.key.endswith('.tar'):
                    tar_file = tarfile.open(fileobj = BytesIO(obj.get()['Body'].read()), mode = 'r')
                    return {
                        key : self.create_tar_iter(tar_File, key) for key in self.filter_keys
                    }

        except StopIteration:
            return None
    
    def get_next(self):
        while True:
            try:
                yield {
                    key : decode(
                        next(self.tar_cache[key])
                    ) for decode, key in zip(self.decodes, self.filter_keys)
                }
            except StopIteration:
                self.tar_cache = self.get_next_tar()
                if self.tar_cache is None:
                    raise StopIteration
                else:
                    continue
    
    def __iter__(self):
        return iter(self.get_next())