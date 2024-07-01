
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
import random
import einops as eo
from transformers import CLIPTextModel, CLIPTokenizer
from common.nn.denoiser import ConditionedRectFlowTransformer, CLIPConditioner
from common.configs import ViTConfig

from common.sampling.diffusion import ode_sampling_cfg
from common.sampling import Text2ImageSampler
from common.utils import dict_to

from common.trainer import Trainer
from common.configs import ProjectConfig, TrainConfig, LoggingConfig

from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os

class CocoDataset(Dataset):
    def __init__(self, root_dir="/home/shahbuland/Documents/neuralnet/coco_data/train2014", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class CocoCollator:
    def __init__(self, tokenizer, image_size = 64, cfg_prob = 0.1):
        self.tokenizer = tokenizer
        self.cfg_prob = 0.1
        self.image_size = image_size
        self.cfg_prob = cfg_prob
    
    def __call__(self, batch):
        # Batch is a list of images
        images = batch
        

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()
        ])

        images = [transform(img) for img in images]
        captions = [""] * len(images)
        images = torch.stack(images)

        tokenizer_out = self.tokenizer(captions, return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        input_ids = tokenizer_out['input_ids']
        attention_mask = tokenizer_out['attention_mask']

        res = {
            "pixel_values" : images,
            "input_ids" : input_ids,
            "attention_mask" : attention_mask
        }

        res = dict_to(res, 'cuda:0')

        return res

if __name__ == "__main__":
    # ============ MODEL INITS ==============
    clip_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(clip_id)
    clip_lm = CLIPTextModel.from_pretrained(clip_id)

    text_encoder = CLIPConditioner(clip_lm, tokenizer, layer_skip = -2, hidden_size = 512)
    text_encoder.cuda()
    neg_emb = text_encoder(text="")

    model_config = ViTConfig(
        n_layers = 12,
        n_heads = 12,
        hidden_size = 768,
        input_shape = (3, 64, 64),
        patching = (8, 8),
        flash = False
    )

    denoiser = ConditionedRectFlowTransformer(
        model_config,
        text_encoder
    )
    denoiser.cuda()

    # ========================================

    # ========= FOR SAMPLING =================

    def model_fn(model, text): # where text is list of strings
        text_features = model.text_encoder(text=text)
        return ode_sampling_cfg(
            model,
            model_config.input_shape,
            text_features,
            50,
            neg_emb
        )

    # define preproc and postproc for gen images
    sampler_prompts = [
        "","","",""
    ]
    sampler_pre = lambda x : x
    sampler_post = lambda x : (x.clamp(0,1).permute(0, 2, 3, 1)*255).byte().detach().cpu().numpy()

    sampler = Text2ImageSampler(
        sampler_prompts,
        sampler_pre,
        sampler_post
    )

    # ======= TRAINER SETUP ==============

    config = ProjectConfig(
        train=TrainConfig(
            batch_size = 16, target_batch = 128,
            epochs = 200,
            save_every = 9999,
            sample_every = 101,
            eval_every = 1000,
            checkpoint_dir = "./mscoco_rft_out",
            train_state_checkpoint = "./trainer_state",
            resume = False
        ),
        logging=LoggingConfig(
            run_name = "b32 long (12, 12, 768) (32p4)",
            wandb_entity = "shahbuland",
            wandb_project = "SGM-coco-debug"
        )
    )

    # ====== DATASET =========

    ds = CocoDataset()
    dc = CocoCollator(tokenizer)

    trainer = Trainer(
        denoiser,
        ds, dc, config=config,
        sampler=sampler,
        model_sample_fn=model_fn
    )

    trainer.train()
