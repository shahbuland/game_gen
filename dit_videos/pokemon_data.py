from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
import einops as eo
import random
import torch

"""
Toy dataset using pokemon from BLIP
"""

class VideoDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.ds = load_dataset("lambdalabs/pokemon-blip-captions")['train']
    
    def __getitem__(self, idx):
        return self.ds[idx]

    def __len__(self):
        return len(self.ds)

class DataCollator:
    def __init__(self, tokenizer, size = 32, target_frames = 100, cfg_prob = 0.1, **kwargs):
        self.tokenizer = tokenizer
        self.image_size = size
        self.target_frames = target_frames
        self.cfg_prob = cfg_prob

    def __call__(self, batch):
        images = [d['image'] for d in batch]
        captions = [d['text'] if random.random() >= self.cfg_prob else "" for d in batch]

        
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()
        ])
        
        images = [transform(img) for img in images]
        images = torch.stack(images)

        tokenizer_out = self.tokenizer(captions, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        input_ids = tokenizer_out['input_ids']
        attention_mask = tokenizer_out['attention_mask']

        # Turn the images into mock videos
        videos = eo.repeat(images, 'b c h w -> b n c h w', n = self.target_frames).contiguous()

        return {
            "pixel_values" : videos,
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "frame_rates" : torch.tensor([10.0]*len(videos))
        }

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    ds = VideoDataset()
    dc = DataCollator(tokenizer)

    loader = DataLoader(ds, collate_fn=dc, batch_size=8, num_workers=0)

    for batch in loader:
        print("Shape of videos: ", batch["pixel_values"].shape)
        print("Shape of input_ids: ", batch["input_ids"].shape)
        print("Frame rates: ", batch["frame_rates"])
