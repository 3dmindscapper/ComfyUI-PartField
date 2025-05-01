import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Install requirements if needed
try:
    import trimesh
    import torch
    import numpy as np
    import lightning
    import yacs
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from scipy.sparse import coo_matrix, csr_matrix
    import matplotlib.pyplot as plt
    import networkx as nx
    import requests
    import tqdm
except ImportError:
    print("Installing PartField requirements...")
    import subprocess
    requirements_file = os.path.join(current_dir, "requirements.txt")
    if os.path.exists(requirements_file):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("Requirements installed successfully.")
    else:
        print("Warning: requirements.txt not found.")

from .nodes import (
    PartFieldModelDownLoader,  # Updated import name
    PartFieldInference,
    PartFieldClustering,
    PartFieldSplitter,
    PartFieldExportParts,
    PartFieldViewer,
)

# Updated node mappings
NODE_CLASS_MAPPINGS = {
    "PartFieldModelDownLoader": PartFieldModelDownLoader, # Updated class name
    "PartFieldInference": PartFieldInference,
    "PartFieldClustering": PartFieldClustering,
    "PartFieldSplitter": PartFieldSplitter,
    "PartFieldExportParts": PartFieldExportParts,
    "PartFieldViewer": PartFieldViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PartFieldModelDownLoader": "PartField Model (Down)Loader", # Updated display name
    "PartFieldInference": "PartField Inference",
    "PartFieldClustering": "PartField Clustering",
    "PartFieldSplitter": "PartField Split Mesh",
    "PartFieldExportParts": "PartField Export Parts",
    "PartFieldViewer": "PartField Mesh Viewer",
}

# Keep version
__version__ = "0.1.0"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__'] 