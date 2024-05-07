# Trainer

Trainer object is located in `common/trainer.py`. The constructor
for the trainer takes, in the following  order:
- A model with a rigidly defined forward method:
    -> Parameters should match 1-to-1 with keys in collator out
    -> Output should either be loss as a single scalar, or a tuple of loss and metrics, where metrics is a dictionary of string keys paired with float values to log to wandb
- A dataset (typically just normal torch/hf datasets, but more tailored implementations are available under `common/data/`, as well as S3 utils)
- A data collator that outputs dictionaries corresponding to model forward function parameters. More tailored collators under `common/data`.
- Optionally an eval dataset for evaluation  
- A config (project config)  
- Optionally, a sampler (see samplers.md) which is used to generate samples with the model whenever the config deems it's time to generate samples (i.e. for gen models)
- A model sampling function that is rigidly defined in the following manner:
    -> Takes model as first argument, and any number of other arguments afterwards
    -> Using the arguments (which are assumed to be given by the sampler) makes and returns samples from the model. Sampler does preprocessing and postprocessing here, so the arguments can be assumed preprocessed and can assume the sampler will handle postprocessing the output of this function

# Adversarial Trainer

- Variant of trainer that is mostly identical, except is assumes that the model has methods `focus_main()` and `focus_disc()` that can be used to freeze/unfreeze the competing adversarial components. Takes a tuple "ratios" as an additional argument to train which controls how often the components are trained relative to eachother

# Project Configs

- Consists of `train` and `logging`, which are `TrainConfig` and `LoggingConfig` respectively
- `TrainConfig` has many parameters that are self explanatory (see `common/configs.py`)
- `LoggingConfig` is for wanb (run name, entity, project), see same script `common/configs.py` for more info 

# Sampling

- There are specific samplers for diff kinds of models (i.e. image->image, text->image, video->video, etc.)
- see `common/sampling/__init__.py` for more details
- Typically, each sampler takes a preprocessor and postprocessor
    -> Preprocessor turns data into model ready inputs
    -> Postprocessor turns model output into something that can be directly given to wandb (i.e. wandb.Image, wandb.Video, etc.)
    -> If they aren't given, we just use simple/common processors
- Samplers return dicts with wandb objects paired with keys, which can directly be logged
