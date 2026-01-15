from src.models import abmil, transmil, transformer, ilra, dftd, clam, rrt, wikg, dsmil
import pathlib
import os
REPO_PATH = str(pathlib.Path(__file__).parent.resolve())  # absolute path to repo root
CONFIG_PATH = os.path.join(REPO_PATH, 'model_configs')
MODEL_SAVE_PATH = os.path.join(REPO_PATH, 'model_weights')

ENCODER_DIM_MAPPING : dict[str, int] = {
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
    'musk': 1024,
    'dino_v3' : 1280
}

MODEL_ENTRYPOINTS = {
    'abmil': (abmil.ABMILModel, abmil.ABMILGatedBaseConfig),
    'transmil': (transmil.TransMILModel, transmil.TransMILConfig),
    'transformer': (transformer.TransformerModel, transformer.TransformerConfig),
    'dftd': (dftd.DFTDModel, dftd.DFTDConfig),
    'clam': (clam.CLAMModel, clam.CLAMConfig),
    'ilra': (ilra.ILRAModel, ilra.ILRAConfig),
    'rrt': (rrt.RRTMILModel, rrt.RRTMILConfig),
    'wikg': (wikg.WIKGMILModel, wikg.WIKGConfig),
    'dsmil': (dsmil.DSMILModel, dsmil.DSMILConfig),
}  