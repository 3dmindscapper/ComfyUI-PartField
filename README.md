## Support the Developer

If you find this custom node useful, consider supporting the developer:

[Buy Me a Coffee](https://buymeacoffee.com/3dmindscaper2000)

## License and Usage Restrictions

This project incorporates code derived from the NVIDIA PartField project. As such, this entire ComfyUI custom node is distributed under the NVIDIA License terms, a copy of which is included in the `LICENSE` file in this repository.

**IMPORTANT:** In accordance with the NVIDIA License (Section 3.3), this software **may only be used for non-commercial research and educational purposes.** It is **not licensed for commercial use.**

Please ensure you comply with these terms. The original NVIDIA PartField code includes its own copyright notices which have been retained within the `partfield` directory.

https://github.com/nv-tlabs/PartField

# ComfyUI-PartField

This is a ComfyUI extension that integrates NVIDIA's PartField model for 3D mesh segmentation.

## Overview

PartField learns a latent space of 3D shapes and parts, which can segment any mesh into semantically meaningful parts. 
This ComfyUI extension makes it easy to:

1. Load a PartField model checkpoint
2. Load and preprocess 3D meshes
3. Extract part features from meshes
4. Cluster mesh parts using KMeans or Agglomerative clustering
5. Export colored segmented meshes

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/ComfyUI-PartField
   ```

2. Install required dependencies:
   ```
   cd ComfyUI-PartField
   pip install -r requirements.txt
   ```

3. Start or restart ComfyUI.

## Model Setup

The extension automatically creates a `models/PartField` directory in your ComfyUI installation. Place your PartField model checkpoints (.ckpt, .pth, etc.) in this directory.
They will automatically appear in the dropdown menu of the `PartField Model (Down)Loader` node. The node can also attempt to download a default model if none are found.

## Usage

The extension provides the following nodes:

### PartField Model (Down)Loader
- Loads a PartField model checkpoint using default configurations. Can automatically download a default model if none are present.
- Inputs:
  - `model_name`: Dropdown to select from available models in `ComfyUI/models/PartField`. Will attempt download if empty.
  - `use_gpu`: Whether to use GPU for inference (if available).
- Outputs:
  - `PARTFIELD_MODEL`: Dictionary containing the loaded model, device, and configuration.

### PartField Inference
- Runs inference on a 3D mesh object to extract per-vertex part features.
- Inputs:
  - `partfield_model`: The output from the `PartField Model (Down)Loader` node.
  - `mesh`: A mesh object (e.g., from a loader node or another PartField node). Must be a `trimesh.Trimesh` or compatible object with `.vertices` and `.faces` attributes.
  - `normalize_mesh`: (Optional) Whether to center and scale the mesh to fit within [-1, 1] before inference (default: True).
- Outputs:
  - `mesh`: The input mesh object (passed through, possibly normalized if Trimesh was created internally).
  - `features`: Dictionary containing the extracted per-vertex part features as a NumPy array (`['features']`).

### PartField Clustering
- Clusters mesh faces based on averaged vertex features and assigns colors.
- Inputs:
  - `mesh`: A mesh object (expects output from `PartField Inference`).
  - `features`: The feature dictionary output from `PartField Inference`.
  - `num_clusters`: Desired number of part clusters.
  - `cluster_method`: Method for clustering (`kmeans` or `agglomerative`).
  - `adjacency_option`: (`agglomerative` only) Method for determining face connectivity (`naive` for shared edges only, `mst` for shared edges + KNN/MST for better connectivity).
  - `seed`: (Optional) Random seed for KMeans reproducibility.
- Outputs:
  - `colored_mesh`: A `trimesh.Trimesh` object with vertex/face colors representing clusters.
  - `preview`: Preview image tensor of the segmented mesh.

### PartField Splitter
- Splits a colored mesh (from `PartField Clustering`) into separate mesh objects based on vertex colors.
- Inputs:
  - `colored_mesh`: The colored `trimesh.Trimesh` object from `PartField Clustering`.
- Outputs:
  - `part_meshes`: A list (`LIST_TRIMESH`) containing individual `trimesh.Trimesh` objects, one for each color/part.

### PartField Export Parts
- Exports a list of mesh parts (from `PartField Splitter`) to individual files inside a subfolder within the main ComfyUI output directory.
- Inputs:
  - `part_meshes`: The `LIST_TRIMESH` output from `PartField Splitter`.
  - `output_folder_name`: Name for the subfolder created within the ComfyUI output directory. If empty, a timestamped name is used.
  - `filename_prefix`: (Optional) Prefix for the individual output filenames inside the folder.
  - `file_format`: (Optional) Format to save the meshes (`glb`, `ply`, `obj`).
- Outputs:
  - `exported_paths`: A list (`LIST_STRING`) of the full paths to the exported files.
  - `preview`: Preview image tensor of the first exported part.

### PartField Viewer
- Displays a preview image of an input Trimesh object.
- Inputs:
  - `colored_mesh`: A `trimesh.Trimesh` object (e.g., from `PartField Clustering`).
  - `multi_view`: (Optional) If True, displays a 2x2 grid of different views instead of a single view (default: False).
- Outputs:
  - `preview`: Preview image tensor of the mesh (single or multi-view grid).

## Example Workflow

1. Place your PartField checkpoint file(s) (e.g., `.ckpt`) in the `ComfyUI/models/PartField` directory (or let the loader node download the default).
2. Load your 3D mesh using a suitable ComfyUI mesh loader node.
3. Use `PartField Model (Down)Loader` to select the model.
4. Connect the mesh loader output and the model loader output to `PartField Inference`.
5. Connect the outputs of `PartField Inference` to `PartField Clustering` and configure clustering parameters.
6. (Optional) Connect the `colored_mesh` from `PartField Clustering` to `PartField Splitter`.
7. (Optional) Connect the `part_meshes` from `PartField Splitter` to `PartField Export Parts` to save individual segments.
8. (Optional) Connect the `colored_mesh` from `PartField Clustering` to `PartField Viewer` to see a single or multi-view preview.
9. View the results in the preview images generated by `PartField Clustering`, `PartField Export Parts`, or `PartField Viewer`.

## Troubleshooting

### "Unable to get model file path" Error
If you encounter this error:

1. Check that your model file exists in the `ComfyUI/models/PartField` directory
2. Check the console for more detailed error messages
3. Check that the `PartField Model (Down)Loader` node successfully downloaded the model if needed.

### Model Loading Issues
If the model is found but doesn't load correctly:

1. Ensure the model file is a valid PartField checkpoint
2. Check if the file is corrupted or incomplete (try re-downloading)
3. Check if you have sufficient GPU memory if using GPU

### Mesh Processing Issues
If mesh processing fails:

1. Ensure the mesh file is valid and has properly defined faces and vertices
2. Try simplifying complex meshes before processing
3. Check that the mesh path is correct if loading from a file

## Getting a Checkpoint

You'll need a PartField model checkpoint to use this extension. The `PartField Model (Down)Loader` node will attempt to download a default one. Alternatively, contact NVIDIA or refer to the original PartField repository for information on obtaining other checkpoints.



## License

See the LICENSE file for details.

## Acknowledgements

This project is based on NVIDIA's PartField model. Please see the original paper for more details:
[PartField: Generalizable 3D Part Segmentation with Radiance Fields](https://nvlabs.github.io/partfield/) 