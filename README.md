# MIL-Lab

**Do Multiple Instance Learning Models Transfer?**, ICML 2025 (Spotlight) <br>
*Daniel Shao, Richard J. Chen, Andrew H. Song, Joel Runevic, Ming Y. Lu, Tong Ding, Faisal Mahmood*

 [Paper](https://arxiv.org/abs/2506.09022) | [HuggingFace](https://huggingface.co/collections/MahmoodLab/feather-6875570e0c755f6c9128a85d) | [Cite](#ack)
 
MIL-Lab provides a standardized library for initializing Multiple Instance Learning (MIL) models, as well as loading models pretrained on a challenging pan-cancer morphological classification task (*PC-108*, 108-way classification) on a Mass General Brigham (MGB) internal dataset.
This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Mass General Brigham. 

### Key Features:

- **Lightweight supervised slide foundation models**: We provide **FEATHER**, *a lightweight supervised slide foundation model* that can easily be finetuned on consumer-grade GPUs, using orders of magnitude less parameters than other self-supervised slide foundation models while achieving competitive performance.
- **Standardized MIL implementation**: Construct numerous MIL methods with a single line of code.
- **Support across encoders**: Load models trained on popular patch foundation models including [UNI](https://huggingface.co/MahmoodLab/UNI), [CONCHv1.5](https://huggingface.co/MahmoodLab/conchv1_5), and [UNIv2](https://huggingface.co/MahmoodLab/UNI2-h).
- **Extensive Benchmarking**: Each model is evaluated on 15+ classification tasks in both morphological and molecular subtyping, with benchmarking against slide foundation models such as [TITAN](https://huggingface.co/MahmoodLab/TITAN), [THREADS](https://arxiv.org/abs/2501.16652), [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath), and [CHIEF](https://github.com/hms-dbmi/CHIEF).

### Updates:
- 07/15/25: Feather-24K (CONCHv1.5) has been integrated into [TRIDENT](https://github.com/mahmoodlab/TRIDENT).
- 07/15/25: MIL-lab is now live!

### In progress:
- MIL implementations for [MambaMIL](https://github.com/isyangshu/MambaMIL), [SI-MIL](https://github.com/bmi-imaginelab/SI-MIL), [FR-MIL](https://github.com/PhilipChicco/FRMIL), [DGR-MIL](https://github.com/ChongQingNoSubway/DGR-MIL)

## Model weights
We have expanded the PC-108 dataset to span 24K slides from patients treated at MGB (from 3K in the manuscript). ABMIL models pretrained with 108-way classification task on 24K slides, termed *FEATHER-24K* are available for the following patch encoders.

| Model    | Patch enc. | Link            | How to load |
|---------------------|--------------|---------------------| -- |
| **FEATHER-24K**      |   CONCHv1.5    | [HF Link](https://huggingface.co/mahmoodlab/abmil.base.conch_v15.pc108-24k) |`create_model('abmil.base.conch_v15.pc108-24k')` |
| **FEATHER-24K**      |   UNIv2    | [HF Link](https://huggingface.co/mahmoodlab/abmil.base.uni_v2.pc108-24k) | `create_model('abmil.base.uni_v2.pc108-24k')` |
| **FEATHER-24K**      |   UNI    | [HF Link](https://huggingface.co/mahmoodlab/abmil.base.uni.pc108-24k) | `create_model('abmil.base.uni.pc108-24k')` |


## Benchmarking against slide foundation models
FEATHER models offer an efficient pretraining alternative for slide foundation model development, which dominantly relies on self-supervised learning and thus requires intensive data and computational resources. Our benchmarks across 15 tasks (T=15) show they achieve competitive performance to current SOTA slide foundation models, while substantially reducing training time, model size, and pretraining data requirements. 

| Model<br>(Patch enc.) | Avg.<br>(T=15) | TCGA<br>(T=10) | EBRAINS<br>(T=2) | BRACS<br>(T=2) | PANDA<br>(T=1) | Num.<br> Params | Num.<br> Pretrain |
|:---|---:|---:|---:|---:|---:|:---|:---|
| **FEATHER-24K**<br>(CONCHv1.5) | 76.2 | 76.7 | 80.1 | 62.6 | 91.3 | 0.9M | 24K |
| **FEATHER-24K**<br>(UNIv2) | 75.8 | 75.3 | 82.7 | 62.5 | 93.5 | 0.9M | 24K |
| **FEATHER-24K**<br>(UNI) | 75.3 | 75.8 | 81.0 | 57.8 | 93.4 | 0.9M | 24K |
| TITAN<br>(CONCHv1.5) | 75.9 | 76.8 | 83.4 | 59.6 | 91.8 | 48.5M | 336K |
| THREADS<br>(CONCHv1.5) | 74.1 | 72.5 | 78.7 | 61.8 | 91.4 | 11.3M | 47K |
| GigaPath<br>(GigaPath) | 72.6 | 72.6 | 79.3 | 54.6 | 94.5 | 86.3M | 171K |
| CHIEF<br>(CTransPath) | 69.8 | 70.5 | 71.0 | 58.4 | 84.2 | 0.9M | 43K |

All of the models are finetuned according to their official recipes. Train time is duration for one epoch of training on EBRAINS (1.2k WSIs) with a single A100 GPU. TCGA task group consists of the molecular subtyping tasks reported in the manuscript. 

## Available MIL models
We provide the list of MIL model implementations available in MIL-Lab, adapted from original implementations. This list will be continuously updated, so stay tuned! 

| Model | Code | Paper | Model Class | Initialization |
|:---|:---|:---|:---|:---|
| ABMIL | [Link](./src/models/abmil.py) | [Link](https://arxiv.org/abs/1802.04712) | `ABMILModel()` | `create_model('abmil')`|
| TransMIL | [Link](./src/models/transmil.py) | [Link](https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html) | `TransMILModel()` | `create_model('transmil')`|
| Transformer | [Link](./src/models/transformer.py) | [Link](https://arxiv.org/abs/1706.03762) | `TransformerModel()` |  `create_model('transformer')`|
| WiKG | [Link](./src/models/wikg.py) | [Link](https://arxiv.org/abs/2403.07719) | `WIKGMILModel()` |  `create_model('wikg')`|
| DFTD | [Link](./src/models/dftd.py) | [Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_CVPR_2022_paper.pdf) | `DFTDModel()` |  `create_model('dftd')`|
| DSMIL| [Link](./src/models/dsmil.py) | [Link](https://arxiv.org/abs/2011.08939) | `DSMILModel()` |  `create_model('dsmil')`|
| ILRA | [Link](./src/models/ilra.py) | [Link](https://openreview.net/pdf?id=01KmhBsEPFO) | `ILRAModel()`|  `create_model('ilra')`|
| RRT | [Link](./src/models/rrt.py) | [Link](https://arxiv.org/abs/2402.17228) | `RRTMILModel()`|  `create_model('rrt')`|
| CLAM | [Link](./src/models/clam.py) | [Link](https://www.nature.com/articles/s41551-020-00682-w) | `CLAMModel()` |  `create_model('clam')`|

 ---

# 🔨 1. Installation
```shell
conda create -n "mil" python=3.9 -y
conda activate mil
git clone https://github.com/mahmoodlab/MIL-Lab.git
cd MIL-Lab
pip install -e .
pip install git+https://github.com/oval-group/smooth-topk  # Required for CLAM
```


# 🔨 2. **Loading a MIL model**

Models are named as `<model_name>.<config>.<encoder>.<pretrain_task>`, with corresponding weights that can be accessed from HuggingFace after [requesting permission](https://huggingface.co/collections/MahmoodLab/feather-6875570e0c755f6c9128a85d). 

*Pretrained* models can be initialized either with a `state_dict` or with `AutoModel` 
```python
# construct the model from src and load the state dict from HuggingFace
create_model('abmil.base.uni.pc108-24k', num_classes=5)

# or with HuggingFace's AutoModel using from_pretrained
create_model('abmil.base.uni.pc108-24k', from_pretrained=True, num_classes=5)
```
> [!Note]
> FEATHER models do not include a classification head. Obtain the appropriate output dimension for your needs by specifying `num_classes`
>

To initialize models with *random weights*, use `create_model` or the underlying model architecture implementations as standalone modules. 
```python
create_model('abmil.base.uni.none', num_classes=5)    # directly specify the task as "none"  

# or as a standalone module
from models.abmil import ABMILModel
from models.dsmil import DSMILModel
from models.transmil import TransMILModel

ABMILModel(in_dim=1024, num_classes=2)
DSMILModel(in_dim=1024, num_classes=2)
TransMILModel(in_dim=1024, num_classes=2)
...
```

> [!NOTE]
> Feeling lazy? `create_model` will also use default values if a shorter name is supplied
>
You can provide varying levels of detail in the model name. Default values of `config=base`, `encoder=uni`, `task=none` will be filled in
```python
# the following models are equivalent
create_model('abmil')
create_model('abmil.base')
create_model('abmil.base.uni')
create_model('abmil.base.uni.none')
```
## Model inference
Inference with an MIL model can be performed as follows:

```python
features = torch.randn(1,100,1024)

model = create_model('abmil')
results_dict, log_dict = model(features, 
                               loss_fn=nn.CrossEntropyLoss(), 
                               label=torch.LongTensor([1]), 
                               return_attention=True,
                               return_slide_feats=True
)
```
**Input** 
- Batch of patch features (`torch.Tensor`) of shape `(batch_size, num_patches, feature_dim)`.
- The patch features can be easily extracted with our sister repo [TRIDENT](https://github.com/mahmoodlab/TRIDENT).

**Args**
- `loss_fn`: Optional loss function for computing loss based on the model output. Required for models with auxiliary losses.
- `label`: Ground truth label. Required if `loss_fn` is supplied. 
- `return_attention`: If True, returns attention scores indicating patch importance, with different definitions across MIL models.
- `return_slide_feats`: If True, returns slide-level features used for classification head.

**Output** 
- `results_dict`
    - `logits`: Output of the model in shape `(batch_size, num_classes)`
    - `loss`: If `label` and `loss_fn` are supplied, then loss will also be included in the output dict.
- `log_dict`: Contains logits and loss as numpy arrays for easier logging
    - `attention`: Predicted attention scores
    - `slide_feats`: Slide-level features
  
**Additional bits** 
- For models which use auxiliary loss, including CLAM and DFTD, the `label` and `loss_fn` arguments are required. Note that models with augmented loss will return both `loss` indicating a weighted loss between `loss_fn()` and the auxiliary loss. The loss from only `loss_fn` can be accessed via `log_dict['cls_loss']`

# 🔨 3. **Customize MIL-Lab**
Users can flexibly introduce new 1) hyperparameter configurations, 2) encoders, and 3) MIL architectures.
> [!NOTE]
> Contributions are welcome! Feel free to create pull requests with additional MIL implementations. Upon review, we can perform PC-108 pretraining on proposed implementations
> 

## New configurations
To create a new set of hyperparameters for your model, you can directly pass in the hyperparameters into create_model. For instance, apply a dropout of 0.3 with a embedding dimension of 256 

```python
create_model('abmil.base.uni.none', dropout=0.3, embed_dim=256, num_classes=2)
```
Alternatively, you can make a new config by creating a yaml file under `model_configs/{model_name}/{config_name}.yaml` and initialize it using this new `name`
```python
create_model('abmil.name.uni.none')
```

## New encoders
The encoder argument is used to infer the feature dimension, `in_dim`. New encoders can be supported by updating the following dict in `builders/_global_mappings.py`
```python
ENCODER_DIM_MAPPING = {
    'uni': 1024,
    'uni_v2': 1536,
    'ctranspath': 768,
    'conch': 512,
    'conch_v15': 768,
    'gigapath': 1536,
    'resnet50': 1024,
    'virchow': 2560,
    'virchow2': 2560,
    'phikon': 768,
    'phikon2': 1024,
    'hoptimus': 1536,
    'hoptimus1': 1536,
    'musk': 1024
}
```

## New MIL architecture

To add a new MIL architecture, follow the checklist below:
- [ ] Add a model class to `models` which inherits from `mil_template.MIL`
- [ ] Implement the forward functions (`forward_features`, `forward_attention`, `forward_head`, and `forward`)
- [ ] Add a config class inherting from `transformers.PretrainedConfig`
- [ ] Add a new config under `model_configs/{model_name}/base.yaml`
- [ ] Update `MODEL_ENTRYPOINTS` within `builders/_global_mappings.py` with a map between `{model_name}` and new class and config.

## Issues

- The preferred mode of communication is via GitHub issues.
- If GitHub issues are inappropriate, email dshao@mit.edu and asong@bwh.harvard.edu

## Funding
This work was funded by NIH NIGMS R35GM138216.

## License and Terms of Use
ⓒ Mahmood Lab. This repository is released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this repository is prohibited and requires prior approval. By downloading any pretrained encoder, you agree to follow the model's respective license.

## Acknowledgements
The project was built on top of amazing repositories such as [Timm](https://github.com/huggingface/pytorch-image-models/), [HuggingFace](https://huggingface.co/docs/datasets/en/index), and open-source contributions for all MIL models from the community. We thank the authors and developers for their contribution. 

## Cite<a id='ack'></a>
If you find our work useful in your research, please cite our paper:

```bibtext
@inproceedings{shao2025do,
    title={Do Multiple Instance Learning Models Transfer?},
    author={Shao, Daniel and Chen, Richard J and Song, Andrew H and Runevic, Joel and Lu, Ming Y. and Ding, Tong and and Mahmood, Faisal},
    booktitle={International conference on machine learning},
    year={2025},
}
```


<img src="_readme/joint_logo.png"> 
