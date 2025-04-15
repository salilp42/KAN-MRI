import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import seaborn as sns
from scipy.stats import zscore
import networkx as nx
from torch_geometric.utils import to_networkx
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class FeatureVisualizer:
    """Class for visualizing model features and internal representations."""
    
    def __init__(self, save_dir: str = 'results/visualizations'):
        """Initialize feature visualizer."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_kernels(self,
                         model: nn.Module,
                         layer_name: str,
                         max_kernels: int = 64,
                         save_name: str = 'kernels.png'):
        """Visualize convolutional kernels."""
        # Get the specified layer
        layer = dict([*model.named_modules()])[layer_name]
        if not isinstance(layer, nn.Conv2d) and not isinstance(layer, nn.Conv3d):
            raise ValueError(f"Layer {layer_name} is not a convolutional layer")
        
        # Get kernels
        kernels = layer.weight.data.cpu().numpy()
        
        if isinstance(layer, nn.Conv3d):
            # For 3D convolutions, take central slice
            kernels = kernels[:, :, kernels.shape[2]//2, :, :]
        
        n_kernels = min(kernels.shape[0], max_kernels)
        n_channels = kernels.shape[1]
        
        # Create subplot grid
        n_cols = 8
        n_rows = int(np.ceil(n_kernels / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
        axes = axes.flatten()
        
        for i in range(n_kernels):
            # Average across channels for visualization
            kernel = np.mean(kernels[i], axis=0)
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())
            
            axes[i].imshow(kernel, cmap='viridis')
            axes[i].axis('off')
        
        # Remove empty subplots
        for i in range(n_kernels, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name)
        plt.close()
    
    def visualize_feature_maps(self,
                             feature_maps: torch.Tensor,
                             layer_name: str,
                             max_features: int = 64,
                             save_name: Optional[str] = None):
        """Visualize feature maps from a layer."""
        feature_maps = feature_maps.cpu().numpy()
        
        if len(feature_maps.shape) == 5:  # 3D feature maps
            feature_maps = feature_maps[:, :, feature_maps.shape[2]//2, :, :]
        
        n_samples = feature_maps.shape[0]
        n_features = min(feature_maps.shape[1], max_features)
        
        for sample_idx in range(n_samples):
            n_cols = 8
            n_rows = int(np.ceil(n_features / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
            axes = axes.flatten()
            
            for i in range(n_features):
                feature_map = feature_maps[sample_idx, i]
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
                
                axes[i].imshow(feature_map, cmap='viridis')
                axes[i].axis('off')
            
            # Remove empty subplots
            for i in range(n_features, len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            if save_name:
                plt.savefig(self.save_dir / f'{save_name}_sample_{sample_idx}.png')
            plt.close()
    
    def visualize_graph_structure(self,
                                graph_data: Union[torch.Tensor, nx.Graph],
                                node_features: Optional[torch.Tensor] = None,
                                edge_weights: Optional[torch.Tensor] = None,
                                save_name: str = 'graph_structure.html'):
        """Visualize graph structure using plotly."""
        if isinstance(graph_data, torch.Tensor):
            # Convert to networkx graph
            G = nx.Graph()
            edges = graph_data.cpu().numpy()
            G.add_edges_from(edges)
        else:
            G = graph_data
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()))
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=10,
                line_width=2
            )
        )
        
        # Add node features if provided
        if node_features is not None:
            node_features = node_features.cpu().numpy()
            if node_features.shape[1] > 1:
                # Use t-SNE for high-dimensional features
                tsne = TSNE(n_components=1)
                node_colors = tsne.fit_transform(node_features).flatten()
            else:
                node_colors = node_features.flatten()
            
            node_trace.marker.color = node_colors
            node_trace.marker.colorscale = 'Viridis'
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=0),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        fig.write_html(self.save_dir / save_name)
    
    def visualize_attention_weights(self,
                                  attention_weights: torch.Tensor,
                                  save_name: str = 'attention_weights.png'):
        """Visualize attention weights."""
        weights = attention_weights.cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights, cmap='viridis', annot=True, fmt='.2f')
        plt.title('Attention Weights')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name)
        plt.close()
    
    def visualize_feature_space(self,
                              features: torch.Tensor,
                              labels: torch.Tensor,
                              method: str = 'tsne',
                              save_name: str = 'feature_space.png'):
        """Visualize high-dimensional feature space."""
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensionality
        reduced_features = reducer.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                            c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'Feature Space Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name)
        plt.close()
    
    def visualize_layer_activations(self,
                                  activations: Dict[str, torch.Tensor],
                                  save_prefix: str = 'layer_activation'):
        """Visualize activations across different layers."""
        for layer_name, activation in activations.items():
            activation = activation.cpu().numpy()
            
            # Compute statistics
            mean_activation = np.mean(activation, axis=0)
            std_activation = np.std(activation, axis=0)
            
            # Plot distribution
            plt.figure(figsize=(10, 6))
            plt.hist(mean_activation.flatten(), bins=50, density=True)
            plt.title(f'Activation Distribution - {layer_name}')
            plt.xlabel('Activation Value')
            plt.ylabel('Density')
            plt.savefig(self.save_dir / f'{save_prefix}_{layer_name}_dist.png')
            plt.close()
            
            # Plot heatmap of mean activations
            plt.figure(figsize=(12, 8))
            sns.heatmap(mean_activation, cmap='viridis')
            plt.title(f'Mean Activation Pattern - {layer_name}')
            plt.savefig(self.save_dir / f'{save_prefix}_{layer_name}_heatmap.png')
            plt.close()
    
    def create_visualization_report(self,
                                  model: nn.Module,
                                  sample_input: torch.Tensor,
                                  save_path: str = 'visualization_report.json'):
        """Create comprehensive visualization report."""
        report = {
            'visualizations': [],
            'model_architecture': str(model),
            'input_shape': list(sample_input.shape)
        }
        
        # Visualize kernels for all convolutional layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                save_name = f'kernels_{name}.png'
                self.visualize_kernels(model, name, save_name=save_name)
                report['visualizations'].append({
                    'type': 'kernels',
                    'layer': name,
                    'path': str(self.save_dir / save_name)
                })
        
        # Save report
        with open(self.save_dir / save_path, 'w') as f:
            import json
            json.dump(report, f, indent=4)
        
        return report
