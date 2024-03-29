from typing import Tuple, List, Dict, Callable, Any

from torch.utils.data import IterableDataset, get_worker_info

import tarfile
from tarfile import TarFile
from PIL import Image
import json
from io import BytesIO
import boto3
import os
import random

def read_from_tar(tar, path):
    file = tar.extractfile(path)
    content = file.read()
    return BytesIO(content)

def default_pil_decode(content : BytesIO):
    return Image.open(content).convert("RGB")

def default_json_decode(content : BytesIO):
    return json.load(content)

def default_txt_decode(content : BytesIO):
    return content.read().decode("utf-8")

def default_mp4_decode(content : BytesIO):
    return content

ext_decode_map = {
    ".json" : default_json_decode,
    ".jpg" : default_pil_decode,
    ".jpeg" : default_pil_decode,
    ".png" : default_pil_decode,
    ".txt" : default_txt_decode,
    ".mp4" : default_mp4_decode
}

class S3Dataset(IterableDataset):
    """
    :param s3_url: URL to S3 dataset
    
    :param filter_keys: Iterable of keys within tar files that we are actually interested in
    (keys = file extensions or common suffixes, ala .jpg or _chosen.jpg)
    :param filter_fn: For each key, we will apply the corresponding filter function. The
    data item will be skipped if it doesn't pass the filter for that key. We will still decode. 
    :param output_keys: If given, use this for the actual dataset items. Useful for skipping items that are only for filtering

    :param decodes: Iterable of functions that can be applied to files obtained from above keys
    (if this is not passed, we can try to guess from keys (i.e. if they're text, or json or jpg)
    :param return_list: By default we return dict with keys mapped to respective data
    If this is not desired, we can instead just return a raw list

    :param offset: Offset dataset iteration within tarfiles? Needed for multiproc. Must provide as
    tuple of (n_processes, process_id)
    :param shard_shuffle: Assumes the top most level in the bucket is 
    :param shard_ids:  
    """
    def __init__(
        self,
        s3_url,
        filter_keys : List[str], filter_fns : Dict[str, Callable[[Any], bool]] = {},
        output_keys : List[str] = None,
        decodes = None, return_list = True,
        offset : Tuple = None,
        shard_shuffle : bool = False,
        shard_ids : List[str] = None,
        
    ):
        s3 = boto3.resource('s3')
        bucket_name, prefix = s3_url.replace("s3://", "").split("/", 1)
        self.bucket = s3.Bucket(bucket_name)
        self.filter_fns = filter_fns

        self.shard_shuffle = shard_shuffle
        self.prefixes = None

        if shard_shuffle:
            # Shuffle will be called on iteration
            self.prefixes = [os.path.join(prefix, shard_id) for shard_id in shard_ids]
            random.shuffle(shard_ids)
            self.objects = None
        else:
            self.objects = iter(self.bucket.objects.filter(Prefix=prefix))

        self.offset = offset

        if decodes is None:
            decodes = []
            for key in filter_keys:
                for ext in ext_decode_map:
                    if key.endswith(ext):
                        decodes.append(ext_decode_map[ext])
                        break

        self.filter_keys = filter_keys
        self.decodes = decodes
        self.return_list = return_list

        if output_keys is None:
            self.output_keys = filter_keys
        else:
            self.output_keys = output_keys

        # Cache is dict (over keys) of iterators over current tar file
        # There is a unique iterator for each key
        self.tar_cache = None

    def shuffle_shards(self, seed : int = None):
        if seed is not None: random.seed(seed)
        random.shuffle(self.prefixes)
        self.prefixes = iter(self.prefixes)

    def get_next_shard(self):
        assert self.shard_shuffle
        self.objects = iter(self.bucket.objects.filter(Prefix=next(self.prefixes)))

    def check_tar_number(self, key) -> int:
        """
        foo/bar/000232.tar -> 232, check if this device should process it
        """
        if self.offset is None:
            return True
        
        n_proc, proc_id = self.offset
        
        num = int(key.split("/")[-1][:-4])

        return num % n_proc == proc_id

    def create_tar_iter(self, tarfile : TarFile, key : str):
        for member in tarfile.getmembers():
            if member.isfile() and member.name.endswith(key):
                yield read_from_tar(tarfile, member.name)
    
    def get_next_tar(self):
        try:
            while True:
                obj = next(self.objects)
                if obj.key.endswith('.tar') and self.check_tar_number(obj.key):
                    tar_file = tarfile.open(fileobj = BytesIO(obj.get()['Body'].read()), mode = 'r')
                    return {
                        key : self.create_tar_iter(tar_file, key) for key in self.filter_keys
                    }

        except StopIteration:
            if self.shard_shuffle:
                try:
                    self.get_next_shard()
                    return self.get_next_tar()
                except StopIteration:
                    return None
            else:
                return None
    
    def get_next(self):
        while True:
            try:
                res = {
                    key : decode(next(self.tar_cache[key])) for decode, key in zip(self.decodes, self.filter_keys)
                }
                skip_this = False
                for key in self.filter_fns:
                    if not self.filter_fns[key](res[key]):
                        skip_this = True # Didn't pass filter, continue
                        break # Can't continue in this for loop, need to do it outside
                if skip_this:
                    continue

                # Only use output keys
                res = {k : res[k] for k in self.output_keys}
                
                if self.return_list:
                    res = list(res.values())
                    if len(res) == 1:
                        res = res[0]
                
                yield res

            except StopIteration:
                self.tar_cache = self.get_next_tar()
                if self.tar_cache is None:
                    raise StopIteration
                else:
                    continue
    
    def __iter__(self):
        if self.shard_shuffle:
            info = get_worker_info()
            if info is not None:
                seed = info.seed
            else:
                seed = None
            self.shuffle_shards(seed)
            self.get_next_shard()
            self.tar_cache = self.get_next_tar()
        
        return iter(self.get_next())