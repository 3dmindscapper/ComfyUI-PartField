import torch
import numpy as np
import trimesh
from PIL import Image
import os
import sys
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
from collections import defaultdict
import lightning.pytorch as pl
from yacs.config import CfgNode as CN
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import einops
from functools import partial
import math
import torch_cluster
import time
import uuid
import io
from plyfile import PlyData, PlyElement
import open3d as o3d
from scipy.spatial import cKDTree
from collections import Counter
import traceback
import requests
from tqdm import tqdm
import folder_paths
import datetime

# --- Utility Functions ---

def create_mesh_preview(mesh, size=(512, 512)):
    """Creates preview image of the colored mesh."""
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"⚠️ Warning: Cannot create preview for non-Trimesh object (type: {type(mesh)}). Returning blank tensor.")
        return torch.zeros((1, size[1], size[0], 3), dtype=torch.float32) # BHWC
    
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print(f"⚠️ Warning: Cannot create preview for empty mesh. Returning blank tensor.")
        return torch.zeros((1, size[1], size[0], 3), dtype=torch.float32) # BHWC
        
    try:
        scene = mesh.scene()
        # Ensure the scene has some content before attempting to render
        if not scene.geometry:
             print(f"⚠️ Warning: Trimesh scene created but has no geometry. Returning blank tensor.")
             return torch.zeros((1, size[1], size[0], 3), dtype=torch.float32)
             
        png_data = scene.save_image(size=size, resolution=None) # Let trimesh handle resolution
        if not png_data:
             print(f"⚠️ Warning: Trimesh scene.save_image returned empty data. Returning blank tensor.")
             return torch.zeros((1, size[1], size[0], 3), dtype=torch.float32)
             
        img = Image.open(io.BytesIO(png_data)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0 # HWC, 0-1 range
        img_tensor = torch.from_numpy(img_np).unsqueeze(0) # Add Batch dim -> BHWC
        return img_tensor
    except ImportError as e:
         # Specific check for pyglet missing, as seen in logs
         if 'pyglet' in str(e):
              print(f"⚠️ Warning: Failed to create mesh preview: {e}. Pyglet might be missing. Try 'pip install pyglet'. Returning blank tensor.")
         else:
              print(f"⚠️ Warning: Failed to create mesh preview due to import error: {e}. Returning blank tensor.")
         return torch.zeros((1, size[1], size[0], 3), dtype=torch.float32) # BHWC
    except Exception as e:
        print(f"⚠️ Warning: Failed to create mesh preview: {str(e)}. Returning blank tensor.")
        # traceback.print_exc() # Uncomment for detailed errors
        return torch.zeros((1, size[1], size[0], 3), dtype=torch.float32) # BHWC

# --- Imports from local 'partfield' directory ---
try:
    # Assumes 'partfield' directory is in the ComfyUI-PartField custom node root
    from .partfield.config import default_argument_parser, setup
    from .partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
    from .partfield.model.triplane import TriplaneTransformer
    from .partfield.model.model_utils import VanillaMLP
    from .partfield.utils import load_mesh_util # Assuming this contains necessary mesh loading
except ImportError as e:
    print(f"Error importing from local 'partfield' directory: {e}")
    print("Please ensure the 'partfield' directory from the original NVIDIA repo is placed inside the 'ComfyUI-PartField' custom_nodes folder.")
    # Provide dummy classes/functions to prevent ComfyUI from breaking on load
    # This allows the user to see the error message and fix the directory structure.
    class TriPlanePC2Encoder(nn.Module): pass
    class TriplaneTransformer(nn.Module): pass
    class VanillaMLP(nn.Module): pass
    def setup(*args, **kwargs): return CN()
    def default_argument_parser(*args, **kwargs):
        import argparse
        return argparse.ArgumentParser()
    def load_mesh_util(*args, **kwargs): return None
    def sample_triplane_feat(*args, **kwargs): return None
    # Raise the error after defining dummies so ComfyUI might still load partially
    # raise ImportError("Could not import necessary modules from the local 'partfield' directory. See console for details.") from e

# --- Default Config Function ---
def get_base_config():
    """Provides a base CfgNode structure matching common PartField defaults."""
    _C = CN()
    _C.seed = 0
    _C.triplane_resolution = 128
    _C.triplane_channels_low = 128
    _C.triplane_channels_high = 512
    _C.use_pvcnnonly = True # Assuming PVCNN only based on common usage
    _C.use_2d_feat = False

    _C.pvcnn = CN()
    _C.pvcnn.point_encoder_type = 'pvcnn'
    _C.pvcnn.use_point_scatter = True # Often True for better quality
    _C.pvcnn.z_triplane_channels = 256 # Matches common checkpoints
    _C.pvcnn.z_triplane_resolution = 128 # Default resolution

    _C.pvcnn.unet_cfg = CN()
    _C.pvcnn.unet_cfg.enabled = True
    _C.pvcnn.unet_cfg.depth = 3
    _C.pvcnn.unet_cfg.start_hidden_channels = 32
    _C.pvcnn.unet_cfg.rolled = True
    _C.pvcnn.unet_cfg.use_3d_aware = True # Check if needed
    _C.pvcnn.unet_cfg.use_initial_conv = False

    # Defaults for TriplaneTransformer (values from original defaults.py)
    # Note: These might be overridden by loaded YAML config
    _C.voxel2triplane = CN()
    _C.voxel2triplane.transformer_dim = 1024
    _C.voxel2triplane.transformer_layers = 6
    _C.voxel2triplane.transformer_heads = 8
    _C.voxel2triplane.triplane_low_res = 32 # Common setting
    _C.voxel2triplane.triplane_high_res = 128 # Matches default triplane_resolution
    _C.voxel2triplane.triplane_dim = 64 # Usually related to sdf_decoder input

    # Add other necessary defaults if your specific checkpoint requires them
    _C.vertex_feature = False # Default for inference often uses face features
    _C.n_point_per_face = 1000 # Example, adjust as needed
    _C.n_sample_each = 10000 # Example, adjust as needed

    return _C

# --- PartField Model Definition (aligns with checkpoint structure) ---
class PartFieldModel(nn.Module):
    """PartField model structure mirroring the training setup for checkpoint loading."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # PVCNN Encoder (imported)
        self.pvcnn = TriPlanePC2Encoder(
            cfg.pvcnn,
            # Let the loader handle device placement
            shape_min=-1.0, # Standard normalization range
            shape_length=2.0,
            use_2d_feat=cfg.use_2d_feat
        )

        # Triplane Transformer (imported)
        # Determine input dim based on PVCNN output and potential concatenation
        # This assumes the pvcnn outputs cfg.pvcnn.z_triplane_channels
        # The original trainer used triplane_channels_low * 2 ? Check model_trainer...
        # Using cfg.pvcnn.z_triplane_channels as input_dim for simplicity, adjust if needed
        transformer_input_dim = cfg.pvcnn.z_triplane_channels

        # Check if the specific TriplaneTransformer used during training had different input
        # The original training script `model_trainer_pvcnn_only_demo.py` uses `input_dim=cfg.triplane_channels_low * 2`
        # Let's use that to match the likely checkpoint structure
        if hasattr(cfg, 'triplane_channels_low'):
             transformer_input_dim = cfg.triplane_channels_low * 2
        else:
             print("Warning: cfg.triplane_channels_low not found, using pvcnn output dim for transformer input.")
             transformer_input_dim = cfg.pvcnn.z_triplane_channels

        self.triplane_transformer = TriplaneTransformer(
            input_dim=transformer_input_dim, # Match training script
            transformer_dim=cfg.voxel2triplane.transformer_dim,
            transformer_layers=cfg.voxel2triplane.transformer_layers,
            transformer_heads=cfg.voxel2triplane.transformer_heads,
            triplane_low_res=cfg.voxel2triplane.triplane_low_res,
            triplane_high_res=cfg.voxel2triplane.triplane_high_res, # Should match triplane_resolution
            triplane_dim=cfg.triplane_channels_high, # <<< FIX: Use triplane_channels_high for output dim
        )

        # SDF Decoder (imported VanillaMLP)
        # The checkpoint expects 'sdf_decoder.layers...', so we use VanillaMLP
        # Determine input dimension (often 64 based on checkpoint errors/structure)
        sdf_input_dim = 64 # Default assumption based on typical split
        # Use the corrected total output dimension for splitting
        total_output_dim = cfg.triplane_channels_high
        # Assume the split logic from training: sdf_dim = 64
        sdf_input_dim = 64
        part_feature_dim = total_output_dim - sdf_input_dim
        if part_feature_dim <= 0:
             print(f"Warning: Calculated non-positive part feature dimension ({part_feature_dim}) using total_output_dim={total_output_dim}. Defaulting sdf_input_dim to 64.")
             sdf_input_dim = 64

        self.sdf_decoder = VanillaMLP(
            input_dim=sdf_input_dim, # Typically 64
            output_dim=1,
            out_activation="tanh", # Common for SDF
            n_neurons=64,          # Match typical training setup
            n_hidden_layers=6      # Match typical training setup
        )

        self.logit_scale = nn.Parameter(torch.tensor([1.0])) # Common parameter

    def forward(self, point_cloud_xyz, point_cloud_feature=None):
        """
        Forward pass mirroring the inference logic.
        Input: point_cloud_xyz (B, N, 3), point_cloud_feature (B, N, F) [Optional, e.g., normals]
        Output: Dictionary containing 'part_features' (B, N, part_dim)
        """
        # 1. Encode Point Cloud to Triplanes using PVCNN
        # Ensure input features are concatenated if provided
        if point_cloud_feature is not None:
            # Verify channel dimensions if needed (referencing TriPlanePC2Encoder logic)
            pc_input = torch.cat([point_cloud_xyz, point_cloud_feature], dim=-1)
        else:
            # Handle case where only XYZ is provided, ensure PVCNN handles it
            # Need to know the expected input channels for PVCNN
            # Assuming PVCNN expects 6 channels (XYZ + Normals) for now
            # If only XYZ is given, we might need padding or specific handling
            if point_cloud_xyz.shape[-1] == 3:
                print("Warning: PVCNN likely expects 6 input channels (XYZ+Normals). Received only 3 (XYZ). Padding might be needed or incorrect.")
                # Placeholder: If PVCNN can handle 3 channels, this works. Otherwise, requires modification.
                pc_input = point_cloud_xyz
            else:
                pc_input = point_cloud_xyz # Assume correct channels provided

        # The pvcnn encoder expects input shape [B, N, C]
        # Output shape [B, 3, C_out, H, W]
        # Pass features separately as expected by TriPlanePC2Encoder.forward
        encoded_triplanes = self.pvcnn(
            point_cloud_xyz=point_cloud_xyz,            # Pass original XYZ
            point_cloud_feature=point_cloud_feature     # Pass original features (or None/dummy)
        )

        # 2. Process Triplanes with Transformer
        # Input shape [B, 3, C_in, H, W], Output shape [B, 3, C_out, H', W']
        processed_triplanes = self.triplane_transformer(encoded_triplanes)

        # 3. Sample features for the input points
        # sample_triplane_feat expects normalized positions [-1, 1]
        # Input point_cloud_xyz assumed to be in this range after preprocessing
        sampled_features = sample_triplane_feat(
            processed_triplanes, # Shape [B, 3, C_out, H', W']
            point_cloud_xyz      # Shape [B, N, 3], assumed normalized to [-1, 1]
        ) # Output shape [B, N, C_out] where C_out = sdf_dim + part_dim

        # 4. Split features into SDF and Part features
        # Using the same split dimensions as during initialization
        sdf_dim = self.sdf_decoder.layers[0].in_features # Get from initialized decoder
        total_dim = sampled_features.shape[-1]
        part_dim = total_dim - sdf_dim

        if part_dim <= 0:
             raise ValueError(f"Calculated non-positive part dimension ({part_dim}) during forward pass. Total sampled dim: {total_dim}, SDF dim: {sdf_dim}")

        sdf_features, part_features = torch.split(
            sampled_features,
            [sdf_dim, part_dim],
            dim=-1
        )

        # (Optional) Run SDF decoder if needed for other tasks, but not for part features
        # sdf_output = self.sdf_decoder(sdf_features)

        # Return part features
        return {'part_features': part_features} # Shape [B, N, part_dim]


# --- ComfyUI Nodes ---

class PartFieldModelDownLoader:
    """Node for loading PartField models and configs, with download capability."""
    @classmethod
    def INPUT_TYPES(s):
        comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(comfy_dir, "models", "PartField")
        default_model_url = "https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt"
        default_model_filename = "model_objaverse.ckpt"
        default_model_path = os.path.join(models_dir, default_model_filename)

        # Ensure models directory exists
        if not os.path.isdir(models_dir):
            print(f"PartField models directory not found. Creating: {models_dir}")
            try:
                os.makedirs(models_dir, exist_ok=True)
            except OSError as e:
                print(f"ERROR: Could not create models directory: {e}")
                # Proceed without download attempt if directory creation fails
                return { "required": { "model_name": (["⚠️ ERROR: Could not create models directory"],), "use_gpu": ("BOOLEAN", {"default": True}), } }

        # Scan for existing models
        model_files = []
        try:
            if os.path.isdir(models_dir):
                 model_files = sorted([f for f in os.listdir(models_dir) if f.lower().endswith(('.ckpt', '.pt', '.pth', '.bin', '.pkl'))])
        except OSError as e:
             print(f"Warning: Could not list models directory {models_dir}: {e}")
             model_files = [] # Ensure it's empty on error

        # Download default model if none exist
        if not model_files:
            print(f"No PartField models found in {models_dir}.")
            if not os.path.exists(default_model_path):
                print(f"Attempting to download default model '{default_model_filename}'...")
                try:
                    response = requests.get(default_model_url, stream=True)
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                    total_size_in_bytes= int(response.headers.get('content-length', 0))
                    block_size = 1024 # 1 Kibibyte

                    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Downloading {default_model_filename}")
                    with open(default_model_path, 'wb') as file:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            file.write(data)
                    progress_bar.close()

                    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                        print("ERROR, something went wrong during download")
                        # Optionally remove partial file: os.remove(default_model_path)
                    else:
                        print(f"Default model downloaded successfully to {default_model_path}")
                        # Re-scan after successful download
                        model_files = sorted([f for f in os.listdir(models_dir) if f.lower().endswith(('.ckpt', '.pt', '.pth', '.bin', '.pkl'))])

                except requests.exceptions.RequestException as e:
                    print(f"ERROR: Failed to download default model: {e}")
                except Exception as e:
                    print(f"ERROR: An unexpected error occurred during download: {e}")
            else:
                print(f"Default model '{default_model_filename}' already exists. Skipping download.")
                # Ensure existing model is picked up if it was the only one
                if not model_files:
                     model_files = [default_model_filename]

        # Check if partfield dir exists and list configs
        partfield_dir = os.path.join(os.path.dirname(__file__), "partfield")
        configs_dir = ""
        if os.path.isdir(partfield_dir):
             configs_dir = os.path.join(partfield_dir, "configs", "final")

        os.makedirs(models_dir, exist_ok=True)

        # Add placeholder if still no models after potential download
        if not model_files:
             model_files = ["⚠️ NO MODELS FOUND - Check logs"] + [" "] # Add space to prevent auto-select issues
        else:
             model_files = [" "] + model_files # Add space to prevent auto-select issues if list isn't empty

        config_files = [" "] # Add space
        try:
            if os.path.isdir(configs_dir):
                 config_files += [os.path.join(configs_dir, f) for f in os.listdir(configs_dir) if f.lower().endswith('.yaml')]
        except OSError as e:
             print(f"Warning: Could not list configs directory {configs_dir}: {e}")

        if len(model_files) == 1: model_files = ["⚠️ NO MODELS FOUND - PLACE IN models/PartField"] + model_files
        if len(config_files) == 1: config_files = ["⚠️ NO CONFIGS FOUND - CHECK partfield/configs/final"] + config_files

        # Set defaults safely
        default_model = model_files[0] # Should be the space

        return {
            "required": {
                "model_name": (model_files, {"default": default_model}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PARTFIELD_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "PartField"

    def load_model(self, model_name, use_gpu):
        if model_name.startswith("⚠️") or not model_name:
             raise ValueError("No valid model selected.")

        comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(comfy_dir, "models", "PartField")
        checkpoint_path = os.path.join(models_dir, model_name)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")

        # Load config - ALWAYS use base defaults now
        cfg = get_base_config() # Start with base defaults
        print("Using base configuration defaults.")

        # Determine device
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load checkpoint
        try:
            # Load CPU first to check keys, prevent OOM
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) # weights_only=False needed for cfg
            state_dict = checkpoint.get('state_dict', checkpoint)
            if state_dict is None:
                 print("Warning: 'state_dict' key not found in checkpoint, using root level keys.")
                 state_dict = checkpoint # Assume checkpoint is the state_dict itself
        except Exception as e:
            print(f"Error loading checkpoint file: {e}")
            raise

        # Instantiate the specific PartFieldModel structure
        # Ensure PartFieldModel class is available here
        try:
             model = PartFieldModel(cfg)
        except Exception as e:
             print(f"Error instantiating PartFieldModel with loaded config: {e}")
             print("Check if the base config and the loaded YAML provide all necessary keys for the model.")
             raise


        # Clean state_dict keys
        cleaned_state_dict = {}
        prefix = "model." # Common prefix in Lightning checkpoints
        for k, v in state_dict.items():
            if k.startswith(prefix):
                cleaned_state_dict[k[len(prefix):]] = v
            else:
                cleaned_state_dict[k] = v

        # Load state dict - use strict=False first for debugging
        print("--- Attempting to load state_dict ---")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
            if missing_keys:
                print(f"⚠️ Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {unexpected_keys}")

            # Check if critical keys are missing or unexpected - adjust logic if needed
            # For now, proceed even with mismatches if strict=False allows it.
            # If strict=True is desired later, these mismatches must be resolved.

        except RuntimeError as e:
             print(f"❌ Error loading state dict: {e}")
             print("This often means the model definition doesn't match the checkpoint structure.")
             print("Ensure the correct config YAML for this checkpoint is selected.")
             # Debug: Print model and checkpoint keys again if error occurs
             print("\n--- Model Structure Keys (Top 50) ---")
             for i, key in enumerate(model.state_dict().keys()):
                  if i >= 50: break
                  print(key)
             print("\n--- Checkpoint Keys (Top 50) ---")
             for i, key in enumerate(cleaned_state_dict.keys()):
                  if i >= 50: break
                  print(key)
             raise e
        except Exception as e:
             print(f"❌ An unexpected error occurred during state dict loading: {e}")
             raise

        # Move model to target device and set to eval mode
        model.to(device)
        model.eval()

        print(f"✅ Model '{model_name}' loaded successfully to {device}.")

        return ({'model': model, 'device': device, 'config': cfg},)


class PartFieldInference:
    """Node for running inference with PartField model"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "partfield_model": ("PARTFIELD_MODEL",),
                "mesh": ("MESH",), # Require direct mesh input
            },
            "optional": {
                "normalize_mesh": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MESH", "FEATURES") # Assuming MESH is a Trimesh object or similar
    RETURN_NAMES = ("mesh", "features")
    FUNCTION = "run_inference"
    CATEGORY = "PartField"

    def preprocess_mesh_for_partfield(self, vertices, faces, normalize=True):
        """Preprocesses mesh vertices (center and scale to [-1, 1])."""
        if not isinstance(vertices, torch.Tensor):
            vertices = torch.tensor(vertices, dtype=torch.float32)
        if not isinstance(faces, torch.Tensor):
            faces = torch.tensor(faces, dtype=torch.int64) # Keep faces as ints

        if normalize:
            center = vertices.mean(dim=0, keepdim=True)
            vertices = vertices - center
            scale = torch.max(torch.norm(vertices, dim=1, p=2))
            vertices = vertices / (scale + 1e-8) # Normalize to fit roughly in [-1, 1] sphere

        return vertices, faces # Return tensors

    def extract_part_features(self, vertices_tensor, model, device):
        """Extracts part features using the loaded PartField model."""
        model.eval() # Ensure eval mode
        # features_list = [] # Not used?
        # batch_size = 1 # Not used directly?
        # num_vertices = vertices_tensor.shape[0] # Not used?

        with torch.no_grad():
             # Add batch dimension
            vertices_batch = vertices_tensor.unsqueeze(0).to(device) # [1, N, 3]

            # Check if the model expects more than 3 channels (XYZ). 
            # Based on TriPlanePC2Encoder init, it hardcodes in_channels=6 for its internal PVCNNEncoder.
            point_features_batch = None
            # Directly check input shape instead of relying on potentially missing attributes.
            if vertices_batch.shape[-1] == 3:
                 print(f"Input has 3 channels (XYZ). Creating dummy zero features for expected 6 channels.")
                 # Create dummy features (e.g., zeros) matching the shape of xyz
                 # Ensure the dummy features have the expected number of channels (3 for normals)
                 expected_feature_channels = 3 # Assuming 6 total expected channels (XYZ + 3 dummy)
                 dummy_shape = list(vertices_batch.shape)
                 dummy_shape[-1] = expected_feature_channels
                 dummy_features = torch.zeros(dummy_shape, dtype=vertices_batch.dtype, device=device)
                 point_features_batch = dummy_features # Shape [1, N, 3]
                 
            # Call the model's forward method with appropriate arguments
            output_dict = model(
                 point_cloud_xyz=vertices_batch,          # Always pass XYZ
                 point_cloud_feature=point_features_batch # Pass dummy features if generated, else None
            )

            part_features = output_dict['part_features'] # Shape [B, N, P]

        # Remove batch dim and move to CPU numpy
        return part_features.squeeze(0).cpu().numpy() # Shape [N, P]


    def run_inference(self, partfield_model, mesh, normalize_mesh=True):
        mesh_source_info = "unknown"
        input_mesh_obj = None # Store the original input mesh
        processed_mesh = None # Store the mesh we'll process (guaranteed Trimesh)

        if mesh is not None:
            print(f"Using direct mesh input of type: {type(mesh)}")
            input_mesh_obj = mesh # Keep the original object
            if isinstance(mesh, trimesh.Trimesh):
                 print("Input is already a Trimesh object.")
                 processed_mesh = mesh # Use it directly
                 mesh_source_info = "direct_trimesh_input"
            elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                 # Attempt to create Trimesh from attributes, converting to NumPy
                 print("Attempting to create Trimesh object from input attributes (.vertices, .faces) via NumPy conversion...")
                 try:
                      # Squeeze potential batch dimension from vertices
                      verts_np = np.array(mesh.vertices).squeeze(0)
                      # Squeeze potential batch dimension from faces
                      faces_np = np.array(mesh.faces).squeeze(0)
                      # Basic shape check
                      if verts_np.ndim != 2 or verts_np.shape[1] != 3:
                           raise ValueError(f"Vertices have incorrect shape: {verts_np.shape}")
                      if faces_np.ndim != 2 or faces_np.shape[1] != 3:
                           raise ValueError(f"Faces have incorrect shape: {faces_np.shape}")

                      processed_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
                      print("Successfully created Trimesh object from attributes after NumPy conversion.")
                      mesh_source_info = f"direct_custom_input ({type(input_mesh_obj)})"
                 except Exception as e:
                      print(f"Conversion Error Details: {e}")
                      # traceback.print_exc() # Uncomment for more debug info
                      raise TypeError(f"Input object has .vertices/.faces, but failed to create Trimesh object even after NumPy conversion.")
            elif isinstance(mesh, (tuple, list)) and len(mesh) == 2:
                 # Attempt to create Trimesh from tuple/list (already converts to numpy)
                 print("Attempting to create Trimesh object from tuple/list input...")
                 try:
                      processed_mesh = trimesh.Trimesh(vertices=np.array(mesh[0]), faces=np.array(mesh[1]), process=False)
                      print("Successfully created Trimesh object from tuple/list.")
                      mesh_source_info = "direct_tuple/list_input"
                 except Exception as e:
                      raise TypeError(f"Input tuple/list could not be converted to Trimesh object: {e}")
            else:
                 raise TypeError(f"Unsupported direct mesh input type: {type(mesh)}. Expected trimesh.Trimesh, object with .vertices/.faces, or (vertices, faces) tuple/list.")
        else:
            # Mesh is required, so if it's None here, it's an error
            raise ValueError("Mesh input is required but received None.")

        # Unpack model data
        model = partfield_model['model']
        device = partfield_model['device']
        cfg = partfield_model['config'] # Get config if needed

        # Validation
        if not hasattr(processed_mesh, 'vertices') or not hasattr(processed_mesh, 'faces') or len(processed_mesh.vertices) == 0 or len(processed_mesh.faces) == 0:
            raise ValueError("Mesh object is invalid or empty after processing input.")
        print(f"Processing mesh (Source: {mesh_source_info}): {len(processed_mesh.vertices)} vertices, {len(processed_mesh.faces)} faces.")

        # Preprocess mesh
        vertices_np = np.array(processed_mesh.vertices, dtype=np.float32)
        faces_np = np.array(processed_mesh.faces, dtype=np.int64)
        vertices_tensor, _ = self.preprocess_mesh_for_partfield(vertices_np, faces_np, normalize=normalize_mesh)

        # Extract features
        print("Extracting PartField features...")
        part_features = self.extract_part_features(vertices_tensor, model, device)
        print(f"Feature extraction complete. Shape: {part_features.shape}")

        # Return the original Trimesh object and the extracted features
        feature_output = {
            'type': 'vertex', # Indicate features are per-vertex
            'features': part_features # Numpy array [N, P]
        }

        # Always return the processed Trimesh object and the features
        return (processed_mesh, feature_output)


class PartFieldClustering:
    """Node for clustering mesh parts using PartField features"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",), # Expects the Trimesh object from Inference node
                "features": ("FEATURES",), # Expects the feature dictionary from Inference node
                "num_clusters": ("INT", {"default": 10, "min": 2, "max": 50}),
                "cluster_method": (["kmeans", "agglomerative"], {"default": "agglomerative"}),
                "adjacency_option": (["naive", "mst"], {"default": "mst"}), # naive=shared edge, mst=knn+mst
            },
            "optional": { # Moved optional inputs here for clarity
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}), # Add seed for kmeans, DEFAULT TO 1
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("colored_mesh", "preview")
    FUNCTION = "cluster_parts"
    CATEGORY = "PartField"

    # --- Helper: construct_face_adjacency_matrix ---
    def construct_face_adjacency_matrix(self, faces, vertices=None, with_knn=True):
        """
        Construct face adjacency matrix based on shared edges.
        Optionally uses KNN+MST to ensure connectivity if `with_knn` is True.
        """
        # (Keep the implementation from previous steps, ensuring NumPy inputs)
        faces_int_array = np.array(faces, dtype=int)
        vertices_np = np.array(vertices, dtype=float) if vertices is not None else None

        if len(faces_int_array.shape) == 3 and faces_int_array.shape[0] == 1:
            faces_int_array = faces_int_array.squeeze(0)
        if vertices_np is not None and len(vertices_np.shape) == 3 and vertices_np.shape[0] == 1:
            vertices_np = vertices_np.squeeze(0)

        num_faces = len(faces_int_array)
        edge_to_faces = defaultdict(list)
        uf = UnionFind(num_faces) # Use local UnionFind helper

        for f_idx, face in enumerate(faces_int_array):
            v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
            edges = [tuple(sorted((v0, v1))), tuple(sorted((v1, v2))), tuple(sorted((v2, v0)))]
            for e in edges:
                edge_to_faces[e].append(f_idx)

        row, col = [], []
        for edge, face_indices in edge_to_faces.items():
            for i in range(len(face_indices)):
                for j in range(i + 1, len(face_indices)):
                    fi, fj = face_indices[i], face_indices[j]
                    row.extend([fi, fj])
                    col.extend([fj, fi])
                    uf.union(fi, fj)

        data = np.ones(len(row), dtype=np.int8)
        face_adjacency = coo_matrix((data, (row, col)), shape=(num_faces, num_faces)).tocsr()

        n_components = sum(1 for i in range(num_faces) if uf.find(i) == i)
        print(f"Mesh components based on shared edges: {n_components}")

        if n_components == 1 or not with_knn or vertices_np is None:
            return face_adjacency

        print("Adding MST edges to connect components...")
        face_centroids = vertices_np[faces_int_array].mean(axis=1)

        k = min(10, num_faces - 1)
        if k <= 0: return face_adjacency # Cannot run KNN

        knn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(face_centroids)
        distances, indices = knn.kneighbors(face_centroids)

        G = nx.Graph()
        G.add_nodes_from(range(num_faces))
        for i in range(num_faces):
            for j_idx, neighbor_idx in enumerate(indices[i]):
                if i != neighbor_idx:
                    G.add_edge(i, neighbor_idx, weight=distances[i][j_idx])

        mst = nx.minimum_spanning_tree(G, weight='weight')
        mst_row, mst_col = [], []
        for u, v in mst.edges():
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                mst_row.extend([u, v])
                mst_col.extend([v, u])

        if mst_row:
            mst_data = np.ones(len(mst_row), dtype=np.int8)
            mst_matrix = coo_matrix((mst_data, (mst_row, mst_col)), shape=(num_faces, num_faces)).tocsr()
            # Combine original adjacency with MST edges
            # Use maximum to avoid double counting, ensure connectivity
            face_adjacency = face_adjacency.maximum(mst_matrix)
            # Alternative: face_adjacency = face_adjacency + mst_matrix, then clip values > 1?

        return face_adjacency.tocsr() # Ensure CSR format

    # --- Main Clustering Function ---
    def cluster_parts(self, mesh, features, num_clusters, cluster_method, adjacency_option,
                     seed=0): # Removed output_dir, save_clustered_mesh
        """Run clustering on mesh features"""
        if mesh is None:
            raise ValueError("Invalid mesh input. Please provide a valid mesh object.")
        if features is None or 'features' not in features or features['features'] is None:
            raise ValueError("Invalid features input.")

        # Extract features (assuming per-vertex from inference)
        vertex_features = features['features'] # Shape [N, P]
        mesh_vertices = np.array(mesh.vertices)
        mesh_faces = np.array(mesh.faces)

        # --- Calculate per-face features by averaging vertex features ---
        try:
            # Ensure face indices are within bounds
            if mesh_faces.max() >= len(vertex_features):
                raise IndexError(f"Max face index ({mesh_faces.max()}) exceeds number of vertices ({len(vertex_features)}).")
            # Average features for vertices of each face
            face_features = vertex_features[mesh_faces].mean(axis=1) # Shape [F, P]
        except IndexError as e:
            raise ValueError(f"Error mapping vertex features to faces: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during face feature calculation: {e}") from e

        # Normalize face features
        features_scaled = face_features / (np.linalg.norm(face_features, axis=-1, keepdims=True) + 1e-8)

        # --- Perform Clustering ---
        labels = None
        start_time = time.time()
        print(f"Starting clustering (Method: {cluster_method}, Clusters: {num_clusters}, Seed: {seed if cluster_method == 'kmeans' else 'N/A'})...")

        if cluster_method == "kmeans":
            # Use the provided seed for random_state
            clustering = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10) 
            labels = clustering.fit_predict(features_scaled)
        elif cluster_method == "agglomerative":
            # Agglomerative doesn't use the seed in this configuration
            use_knn = (adjacency_option == "mst")
            print(f"Constructing adjacency matrix (Option: {adjacency_option}, Use KNN: {use_knn})...")
            adj_matrix = self.construct_face_adjacency_matrix(
                mesh_faces, mesh_vertices, with_knn=use_knn
            )
            print("Adjacency matrix constructed. Starting Agglomerative Clustering...")
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                connectivity=adj_matrix,
                linkage='ward' # Common linkage method
            )
            labels = clustering.fit_predict(features_scaled)
        else:
            raise ValueError(f"Unsupported cluster_method: {cluster_method}")

        print(f"Clustering finished in {time.time() - start_time:.2f} seconds.")

        # Ensure labels are 1D array
        labels = np.array(labels, dtype=int).squeeze()

        # --- Create colored mesh object and preview ---
        colored_mesh_obj = None
        preview_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32) # Default blank

        # Always create the colored mesh in memory for downstream nodes
        try:
            print("Assigning colors to mesh based on labels...")
            # Get colormap
            unique_labels = np.unique(labels)
            num_unique = len(unique_labels)
            min_label, max_label = unique_labels.min(), unique_labels.max()
            cmap_range = max(num_unique, max_label + 1)
            colormap = plt.cm.get_cmap("tab20", cmap_range)
            label_to_color = {
                label: (np.array(colormap(label)[:3]) * 255).astype(np.uint8)
                for label in unique_labels
            }
            
            # Assign face colors
            face_colors_list = []
            default_color = np.array([128, 128, 128, 255], dtype=np.uint8)
            for label in labels:
                color = label_to_color.get(label, default_color[:3])
                face_colors_list.append(np.append(color, 255))
            face_colors = np.array(face_colors_list, dtype=np.uint8)
            
            # Assign vertex colors (first face wins approach)
            vertex_colors = np.full((len(mesh_vertices), 4), [128, 128, 128, 255], dtype=np.uint8)
            for face_idx, face in enumerate(mesh_faces):
                face_color = face_colors[face_idx]
                for vertex_idx in face:
                    if vertex_idx < len(vertex_colors): # Bounds check
                        if np.array_equal(vertex_colors[vertex_idx], [128, 128, 128, 255]):
                            vertex_colors[vertex_idx] = face_color
            default_vert_color = np.array([255, 255, 255, 255], dtype=np.uint8)
            mask_grey = np.all(vertex_colors == [128, 128, 128, 255], axis=1)
            vertex_colors[mask_grey] = default_vert_color

            # Create Trimesh object with colors
            colored_mesh_obj = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces,
                                               face_colors=face_colors, vertex_colors=vertex_colors)
            print("Colored mesh object created.")

            # Generate preview from the colored mesh object
            if colored_mesh_obj:
                print("Generating preview...")
                preview_tensor = create_mesh_preview(colored_mesh_obj)
            else:
                print("Skipping preview generation as colored mesh creation failed.")
                
        except Exception as e:
            print(f"⚠️ Error creating colored mesh object or preview: {str(e)}")
            traceback.print_exc() # Print traceback for debugging
            # Create a dummy mesh if coloring failed but geometry exists
            if colored_mesh_obj is None and len(mesh_vertices) > 0 and len(mesh_faces) > 0:
                 colored_mesh_obj = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)

        print(f"✅ Clustering node complete.")
        # Return the Trimesh object (colored or original) and the preview image tensor
        return (colored_mesh_obj, preview_tensor)

# --- Node to Split Mesh by Color ---
class PartFieldSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "colored_mesh": ("TRIMESH",), # Expects Trimesh object from Clustering
            }
        }

    # Define a custom return type for the list of meshes
    RETURN_TYPES = ("LIST_TRIMESH",)
    RETURN_NAMES = ("part_meshes",)
    FUNCTION = "split_mesh"
    CATEGORY = "PartField"

    def split_mesh(self, colored_mesh):
        if not isinstance(colored_mesh, trimesh.Trimesh):
            raise TypeError(f"Input must be a trimesh.Trimesh object, got {type(colored_mesh)}")
        
        if not hasattr(colored_mesh, 'visual') or not hasattr(colored_mesh.visual, 'vertex_colors'):
             raise ValueError("Input mesh does not have vertex colors. Ensure the clustering node ran successfully.")

        vertices = colored_mesh.vertices
        faces = colored_mesh.faces
        vertex_colors = colored_mesh.visual.vertex_colors

        if len(vertices) == 0 or len(faces) == 0:
            print("Warning: Input mesh has no vertices or faces. Returning empty list.")
            return ([],)

        if len(vertex_colors) != len(vertices):
             raise ValueError(f"Vertex count ({len(vertices)}) does not match vertex color count ({len(vertex_colors)}).")

        print(f"Splitting mesh with {len(vertices)} vertices, {len(faces)} faces.")

        # Group faces by the color of their first vertex
        # Using tuples for colors to make them hashable dict keys
        color_to_faces = defaultdict(list)
        for i, face in enumerate(faces):
            first_vertex_index = face[0]
            if first_vertex_index >= len(vertex_colors):
                 print(f"Warning: Face {i} references vertex {first_vertex_index}, but only {len(vertex_colors)} colors exist. Skipping face.")
                 continue
            # Convert numpy array color to tuple (ignore alpha for grouping if present)
            color_tuple = tuple(vertex_colors[first_vertex_index][:3]) 
            color_to_faces[color_tuple].append(i)

        part_meshes = []
        print(f"Found {len(color_to_faces)} unique color groups (parts).")

        # Create a new mesh for each color group
        for color, face_indices in color_to_faces.items():
            if not face_indices:
                continue # Skip if no faces for this color
            
            # Extract the faces for this part
            part_faces = faces[face_indices]
            
            # Create the submesh
            # Trimesh handles vertex subsetting automatically
            try:
                submesh = trimesh.Trimesh(vertices=vertices, faces=part_faces, process=False) 
                # Optional: Assign a uniform color to the new part for clarity
                # submesh.visual.face_colors = list(color) + [255] # Assign the group color
                part_meshes.append(submesh)
                print(f"  Created part with {len(submesh.faces)} faces for color {color}.")
            except Exception as e:
                 print(f"Warning: Failed to create submesh for color {color}. Error: {e}")

        print(f"Successfully created {len(part_meshes)} separate mesh parts.")
        # Return the list of Trimesh objects within a tuple
        return (part_meshes,)


# --- Node to Export List of Meshes ---
class PartFieldExportParts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "part_meshes": ("LIST_TRIMESH",), # Expects list of Trimesh objects from Splitter
                "output_folder_name": ("STRING", {"default": "partfield_parts"}), # New required input
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "partfield_part"}),
                "file_format": (["glb", "ply", "obj"], {"default": "glb"}),
            }
        }

    RETURN_TYPES = ("LIST_STRING", "IMAGE")
    RETURN_NAMES = ("exported_paths", "preview")
    FUNCTION = "export_parts"
    CATEGORY = "PartField"

    def export_parts(self, part_meshes, output_folder_name, filename_prefix="partfield_part", file_format="glb"):
        preview_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32) # Default blank preview
        exported_paths = []

        if not isinstance(part_meshes, list):
            print(f"Warning: Input must be a list of Trimesh objects, got {type(part_meshes)}. Returning empty list and blank preview.")
            return (exported_paths, preview_tensor)

        if not part_meshes:
            print("Warning: Input list of meshes is empty. Nothing to export.")
            return (exported_paths, preview_tensor)

        # --- Determine Output Directory ---
        base_output_dir = folder_paths.get_output_directory()

        # Sanitize folder name or use timestamp
        if not output_folder_name or output_folder_name.strip() == "":
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            final_folder_name = f"partfield_{timestamp}"
            print(f"Output folder name is empty, using timestamp: {final_folder_name}")
        else:
            # Basic sanitization: replace spaces, remove leading/trailing whitespace
            # Remove potentially problematic characters for folder names
            sanitized_name = output_folder_name.strip().replace(" ", "_")
            final_folder_name = "".join(c for c in sanitized_name if c.isalnum() or c in ('_', '-')).rstrip()
            if not final_folder_name: # Handle case where sanitization results in empty string
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                final_folder_name = f"partfield_{timestamp}"
                print(f"Sanitized folder name was empty, using timestamp: {final_folder_name}")
            else:
                print(f"Using sanitized output folder name: {final_folder_name}")

        # --- DEBUG PRINTS ---
        print(f"[DEBUG] Base Output Dir: {base_output_dir}")
        print(f"[DEBUG] Final Folder Name: {final_folder_name}")
        # --------------------

        # Construct the full path for the subfolder
        output_dir_subfolder = os.path.join(base_output_dir, final_folder_name)

        # --- DEBUG PRINTS ---
        print(f"[DEBUG] Attempting to create folder: {output_dir_subfolder}")
        # --------------------

        # Ensure output subfolder directory exists
        try:
             os.makedirs(output_dir_subfolder, exist_ok=True)
             print(f"Exporting {len(part_meshes)} mesh parts to subfolder: '{output_dir_subfolder}'")
        except OSError as e:
             print(f"Error creating output directory '{output_dir_subfolder}': {e}. Cannot export parts.")
             return ([], preview_tensor)

        num_digits = len(str(len(part_meshes))) # For padding filenames

        for i, mesh_part in enumerate(part_meshes):
            if not isinstance(mesh_part, trimesh.Trimesh):
                print(f"Warning: Item at index {i} is not a Trimesh object (type: {type(mesh_part)}). Skipping.")
                continue

            # Construct filename
            part_filename = f"{filename_prefix}_{i:0{num_digits}d}.{file_format}"
            # Construct the full path including the subfolder for the current part
            output_path = os.path.join(output_dir_subfolder, part_filename) # Use the created subfolder path

            try:
                mesh_part.export(output_path)
                exported_paths.append(output_path)
                print(f"  Exported: {output_path}")
                # Generate preview from the first successfully exported part
                if i == 0:
                    print("Generating preview from first part...")
                    preview_tensor = create_mesh_preview(mesh_part)
            except Exception as e:
                print(f"Error exporting part {i} to {output_path}: {e}")

        print(f"Finished exporting parts.")
        return (exported_paths, preview_tensor)

# --- UnionFind Helper Class (needed for adjacency matrix) ---
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]: self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]: self.parent[rootX] = rootY
            else: self.parent[rootY] = rootX; self.rank[rootX] += 1
            return True
        return False

# --- Node: PartField Viewer ---
class PartFieldViewer:
    """Node to display a preview of a Trimesh object, optionally showing multiple views."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "colored_mesh": ("TRIMESH",), # Expects Trimesh object (e.g., from Clustering)
            },
            "optional": {
                 "multi_view": ("BOOLEAN", {"default": False}), # Option for multi-view grid
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "view_mesh"
    CATEGORY = "PartField"

    def view_mesh(self, colored_mesh, multi_view=False):
        """Generates a single or multi-view preview image for the input mesh."""
        if not isinstance(colored_mesh, trimesh.Trimesh):
            print(f"⚠️ Warning: Input is not a Trimesh object (type: {type(colored_mesh)}). Returning blank tensor.")
            return (torch.zeros((1, 512, 512, 3), dtype=torch.float32),)

        if not multi_view:
            print("Generating single preview for viewer node...")
            # Use the existing global helper function for consistency
            preview_tensor = create_mesh_preview(colored_mesh)
            return (preview_tensor,)
        else:
            print("Generating multi-view preview grid (2x2)...")
            try:
                if len(colored_mesh.vertices) == 0 or len(colored_mesh.faces) == 0:
                     print(f"⚠️ Warning: Cannot create multi-view preview for empty mesh. Returning blank tensor.")
                     return (torch.zeros((1, 512, 512, 3), dtype=torch.float32),) # BHWC

                scene = colored_mesh.scene()
                if not scene.geometry:
                     print(f"⚠️ Warning: Trimesh scene created but has no geometry for multi-view. Returning blank tensor.")
                     return (torch.zeros((1, 512, 512, 3), dtype=torch.float32),)

                # Define target size for each small view (e.g., 256x256 for a 2x2 grid in 512x512)
                view_size = (256, 256)
                grid_dim = 2 # 2x2 grid
                total_size = (view_size[0] * grid_dim, view_size[1] * grid_dim)
                scene.camera.resolution = view_size

                # Auto-set camera distance based on scene size
                scene.camera.z_far = max(10.0, scene.scale * 5.0) # Ensure far plane is adequate
                distance = scene.scale * 1.8 # Heuristic distance - Decreased multiplier to zoom in
                center = scene.centroid # Focus on the center

                # Define 4 camera angles (degrees) for the 2x2 grid
                # [Pitch, Yaw, Roll] - Adjust as needed for desired views
                angles_deg = [
                    [0, 0, 0],       # Front (default)
                    [0, 90, 0],      # Right side (Yaw 90)
                    [90, 0, 0],      # Top (Pitch 90)
                    [45, 45, 0]      # Angled (Pitch 45, Yaw 45)
                ]

                # Create a blank canvas for the grid
                grid_image_pil = Image.new('RGB', total_size, (255, 255, 255)) # White background
                view_images_pil = [] # Store individual view PIL images

                for i, ang_deg in enumerate(angles_deg):
                    if len(view_images_pil) >= grid_dim * grid_dim: break # Ensure we don't render more than needed
                    print(f"  Rendering view {i+1}/{len(angles_deg)} (Angles: {ang_deg})...")

                    try:
                        # Set camera position for this specific view
                        angles_rad = np.radians(ang_deg)
                        # Use set_camera which handles transforms based on angles, distance, center
                        scene.set_camera(angles=angles_rad, distance=distance, center=center)

                        # Render the scene for the current camera view
                        # Use a non-white background during render to differentiate failure vs white object
                        png_data = scene.save_image(background=[200, 200, 200, 255]) # Light grey background
                        
                        if png_data:
                            img = Image.open(io.BytesIO(png_data)).convert("RGB")
                            # Resize just in case save_image didn't respect resolution exactly
                            img = img.resize(view_size) 
                            view_images_pil.append(img)
                        else:
                            print(f"    Warning: scene.save_image returned empty data for view {i+1}.")
                            # Add a placeholder (e.g., dark grey) if render fails
                            view_images_pil.append(Image.new('RGB', view_size, (50, 50, 50)))
                    except Exception as e_render:
                        print(f"    Error rendering view {i+1}: {e_render}")
                        # Add a placeholder (e.g., red) on error
                        view_images_pil.append(Image.new('RGB', view_size, (255, 0, 0)))

                # Assemble the grid
                for i, img in enumerate(view_images_pil):
                    row = i // grid_dim
                    col = i % grid_dim
                    paste_x = col * view_size[0]
                    paste_y = row * view_size[1]
                    grid_image_pil.paste(img, (paste_x, paste_y))
                
                # Convert final grid PIL image to numpy array -> torch tensor
                grid_np = np.array(grid_image_pil).astype(np.float32) / 255.0 # HWC, 0-1 range
                preview_tensor = torch.from_numpy(grid_np).unsqueeze(0) # Add Batch dim -> BHWC

                print("Multi-view grid generated.")
                return (preview_tensor,)

            except ImportError as e_imp:
                 if 'pyglet' in str(e_imp):
                      print(f"⚠️ Error generating multi-view: {e_imp}. Pyglet might be missing. Try 'pip install pyglet'. Returning blank tensor.")
                 else:
                      print(f"⚠️ Error generating multi-view due to import error: {e_imp}. Returning blank tensor.")
                 return (torch.zeros((1, total_size[1], total_size[0], 3), dtype=torch.float32),) # Return blank of correct total size
            except Exception as e:
                print(f"⚠️ Error generating multi-view preview: {str(e)}")
                # traceback.print_exc() # Uncomment for detailed errors
                return (torch.zeros((1, total_size[1], total_size[0], 3), dtype=torch.float32),) # Return blank of correct total size

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "PartFieldModelDownLoader": PartFieldModelDownLoader,
    "PartFieldInference": PartFieldInference,
    "PartFieldClustering": PartFieldClustering,
    "PartFieldSplitter": PartFieldSplitter,
    "PartFieldExportParts": PartFieldExportParts,
    "PartFieldViewer": PartFieldViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PartFieldModelDownLoader": "PartField Model (Down)Loader",
    "PartFieldInference": "PartField Inference",
    "PartFieldClustering": "PartField Clustering",
    "PartFieldSplitter": "PartField Split Mesh",
    "PartFieldExportParts": "PartField Export Parts",
    "PartFieldViewer": "PartField Mesh Viewer",
}