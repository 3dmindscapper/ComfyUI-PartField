import torch
import numpy as np
import os
import sys
import trimesh

def preprocess_mesh_for_partfield(vertices, faces, normalize=True):
    """
    Preprocess a mesh into the format required for PartField inference
    
    Args:
        vertices: Vertex array [N, 3] (tensor or numpy)
        faces: Face array [F, 3] (tensor or numpy)
        normalize: Whether to normalize the mesh to unit sphere
        
    Returns:
        vertices: Tensor of shape [N, 3]
        faces: Tensor of shape [F, 3]
    """
    # Convert to tensor if not already
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(faces, torch.Tensor):
        faces = torch.tensor(faces, dtype=torch.int64)
    
    # Ensure vertices have shape [N, 3]
    if len(vertices.shape) == 1:
        vertices = vertices.reshape(-1, 3)
    elif len(vertices.shape) > 2:
        vertices = vertices.reshape(-1, 3)
    
    # Normalize mesh to unit sphere
    if normalize:
        # Center mesh
        center = vertices.mean(dim=0, keepdim=True)
        vertices = vertices - center
        
        # Scale to unit sphere
        scale = torch.max(torch.norm(vertices, dim=1))
        vertices = vertices / (scale + 1e-8)
    
    return vertices, faces

def extract_part_features(vertices, faces, model, device):
    """
    Extract part features from a mesh using PartField model
    
    Args:
        vertices: Vertex array [N, 3] (tensor)
        faces: Face array [F, 3] (tensor)
        model: PartField model
        device: Torch device
        
    Returns:
        part_features: Feature tensor [1, 448]
    """
    with torch.no_grad():
        # Process through transformer
        vertices = vertices.to(device)
        object_tensor = model.triplane_transformer(vertices)
        
        # Split features - first 64 for SDF, rest for part segmentation
        sdf_features, part_features = torch.split(
            object_tensor, 
            [64, object_tensor.shape[-1] - 64], 
            dim=-1
        )
        
    return part_features

def cluster_parts(mesh, features, num_clusters, cluster_method="kmeans", export_path=None):
    """
    Cluster mesh parts based on extracted features
    
    Args:
        mesh: Trimesh mesh object
        features: Feature tensor [1, 448]
        num_clusters: Number of clusters to generate
        cluster_method: Method to use for clustering ("kmeans" or "agglomerative")
        export_path: Path to export the colored mesh to (optional)
        
    Returns:
        colored_mesh: Mesh with face colors based on clustering
        labels: Cluster labels for each face
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans
    import matplotlib.pyplot as plt
    
    # Normalize features
    features_scaled = features / np.linalg.norm(features, axis=-1, keepdims=True)
    
    # Create face adjacency matrix for agglomerative clustering
    if cluster_method == "agglomerative":
        from scipy.sparse import coo_matrix
        
        # For agglomerative clustering, we need to construct adjacency matrix
        adjacency = construct_face_adjacency_matrix(mesh.faces)
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=num_clusters,
            connectivity=adjacency
        )
    else:
        # KMeans clustering
        clustering = KMeans(n_clusters=num_clusters, random_state=0)
    
    # Fit and predict
    labels = clustering.fit_predict(features_scaled)
    
    # Generate colors for visualization
    colormap = plt.cm.get_cmap("tab20", num_clusters)
    face_colors = (np.array([colormap(i)[:3] for i in labels]) * 255).astype(np.uint8)
    
    # Create colored mesh
    colored_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        face_colors=face_colors
    )
    
    # Export colored mesh
    if export_path:
        colored_mesh.export(export_path)
    
    return colored_mesh, labels

def construct_face_adjacency_matrix(faces):
    """
    Construct face adjacency matrix based on shared edges
    
    Args:
        faces: Face array [F, 3]
        
    Returns:
        adjacency: Sparse adjacency matrix of faces
    """
    from collections import defaultdict
    from scipy.sparse import coo_matrix
    
    # Convert to numpy if tensor
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    
    num_faces = len(faces)
    edge_to_faces = defaultdict(list)
    
    # Build edge to face mapping
    for f_idx, face in enumerate(faces):
        # Standard triangle face - get three edges
        if len(face) >= 3:
            v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
            
            edges = [
                tuple(sorted((v0, v1))),
                tuple(sorted((v1, v2))),
                tuple(sorted((v2, v0)))
            ]
            
            for e in edges:
                edge_to_faces[e].append(f_idx)
    
    # Build adjacency matrix from edge-face mapping
    row = []
    col = []
    for edge, face_indices in edge_to_faces.items():
        for i in range(len(face_indices)):
            for j in range(i + 1, len(face_indices)):
                fi = face_indices[i]
                fj = face_indices[j]
                row.extend([fi, fj])
                col.extend([fj, fi])
    
    data = np.ones(len(row), dtype=np.int8)
    return coo_matrix((data, (row, col)), shape=(num_faces, num_faces)).tocsr() 