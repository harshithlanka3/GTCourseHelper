#!/usr/bin/env python3
"""
3D Embedding Visualization Script

This script loads processed course data and creates an interactive 3D visualization
of course embeddings using dimensionality reduction (PCA, UMAP, or t-SNE) and Plotly.

DIMENSIONALITY REDUCTION EXPLAINED:
- Original embeddings: 384-dimensional vectors from sentence-transformers
- Goal: Reduce to 3D for visualization
- Methods:
  * PCA (linear): Fast, but only explains ~15% variance (loses 85% of information)
  * UMAP (non-linear): Better preserves local neighborhoods, recommended default
  * t-SNE (non-linear): Excellent clusters, but slower and loses global structure

Usage:
    python visualizations/plot_embeddings_3d.py [--data-path DATA_PATH] [--color-by COLOR_BY] [--output OUTPUT]
    
Options:
    --data-path: Path to processed pickle file (default: data/202508_processed.pkl)
    --color-by: Color scheme - 'department' or 'graduate' (default: department)
    --output: Output HTML file path (default: visualizations/embedding_3d.html)
    --limit: Limit number of courses to visualize (default: None, show all)
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path to import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(data_path):
    """Load processed course data and filter out missing embeddings."""
    print(f"Loading data from {data_path}...")
    df = pd.read_pickle(data_path)
    print(f"Loaded {len(df)} courses")
    
    # Filter out rows with missing embeddings
    initial_count = len(df)
    df = df[df['embedding'].notna()].copy()
    df = df[df['embedding'].apply(lambda x: x is not None and len(x) > 0)].copy()
    
    if len(df) < initial_count:
        print(f"Filtered out {initial_count - len(df)} courses with missing embeddings")
    
    print(f"Using {len(df)} courses with valid embeddings")
    return df


def analyze_pca_variance(embeddings, max_components=50):
    """
    Analyze how many PCA components are needed to explain different amounts of variance.
    This helps understand the information loss when reducing to 3D.
    """
    print("\n" + "="*60)
    print("PCA VARIANCE ANALYSIS")
    print("="*60)
    
    # Fit PCA with many components to see variance distribution
    n_components = min(max_components, embeddings.shape[1], embeddings.shape[0] - 1)
    pca_full = PCA(n_components=n_components, random_state=42)
    pca_full.fit(embeddings)
    
    explained_var = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Find components needed for common thresholds
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    print(f"\nOriginal embedding dimension: {embeddings.shape[1]}")
    print(f"Number of samples: {embeddings.shape[0]}")
    print(f"\nVariance explained by top components:")
    print(f"  Top 3 components: {cumulative_var[2]:.2%} (THIS IS WHAT WE'RE USING)")
    print(f"  Top 5 components: {cumulative_var[4]:.2%}")
    print(f"  Top 10 components: {cumulative_var[9]:.2%}")
    print(f"  Top 20 components: {cumulative_var[19]:.2%}")
    
    print(f"\nComponents needed to explain variance thresholds:")
    for threshold in thresholds:
        # Find first index where cumulative variance >= threshold
        mask = cumulative_var >= threshold
        if np.any(mask):
            n_needed = np.argmax(mask) + 1
            print(f"  {threshold:.0%} variance: {n_needed} components")
        else:
            print(f"  {threshold:.0%} variance: >{len(cumulative_var)} components (not reached)")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print("PCA is a LINEAR transformation that finds directions of maximum variance.")
    print("When reducing from 384D → 3D, we're only keeping the top 3 principal components.")
    if cumulative_var[2] < 0.20:
        print(f"\n⚠️  WARNING: Only {cumulative_var[2]:.1%} variance explained!")
        print(f"   This means we're losing ~{1-cumulative_var[2]:.1%} of the information.")
        print("   Consider using UMAP or t-SNE for better local structure preservation.")
    print("="*60 + "\n")


def reduce_to_3d(embeddings, method='umap', show_diagnostics=True):
    """
    Reduce embeddings to 3D using PCA, UMAP, or t-SNE.
    
    Args:
        embeddings: numpy array of embeddings (n_samples, n_features)
        method: 'pca', 'umap', or 'tsne' (default: 'umap')
        show_diagnostics: Whether to show detailed variance analysis for PCA
    
    Returns:
        tuple: (reduced array of shape (n_samples, 3), method_name for labels)
    """
    print(f"\n{'='*60}")
    print(f"REDUCING {embeddings.shape[1]}D EMBEDDINGS → 3D using {method.upper()}")
    print(f"{'='*60}\n")
    
    if method == 'pca':
        if show_diagnostics:
            analyze_pca_variance(embeddings)
        
        pca = PCA(n_components=3, random_state=42)
        reduced = pca.fit_transform(embeddings)
        explained_var = pca.explained_variance_ratio_
        cumulative_var = sum(explained_var)
        
        print(f"PCA Results:")
        print(f"  Component 1 variance: {explained_var[0]:.2%}")
        print(f"  Component 2 variance: {explained_var[1]:.2%}")
        print(f"  Component 3 variance: {explained_var[2]:.2%}")
        print(f"  Total explained: {cumulative_var:.2%}")
        print(f"  Information lost: {1-cumulative_var:.2%}")
        print(f"\n  ⚠️  PCA is LINEAR - good for global structure, but loses local neighborhoods")
        
        return reduced, 'PCA'
        
    elif method == 'umap':
        try:
            import umap
            print("UMAP (Uniform Manifold Approximation and Projection):")
            print("  - NON-LINEAR dimensionality reduction")
            print("  - Preserves LOCAL neighborhood structure better than PCA")
            print("  - Better for visualization when data has non-linear relationships")
            print("  - Typically produces more interpretable clusters\n")
            
            print("Fitting UMAP (this may take a minute for large datasets)...")
            reducer = umap.UMAP(
                n_components=3, 
                random_state=42, 
                n_neighbors=15,  # Balance between local/global structure
                min_dist=0.1,    # Controls how tightly points cluster
                metric='cosine'   # Good for embeddings
            )
            reduced = reducer.fit_transform(embeddings)
            print("✓ UMAP reduction complete")
            print("  UMAP preserves local neighborhoods, so similar courses should cluster together")
            
            return reduced, 'UMAP'
        except ImportError:
            print("⚠️  UMAP not available. Install with: pip install umap-learn")
            print("Falling back to PCA...\n")
            return reduce_to_3d(embeddings, method='pca', show_diagnostics=show_diagnostics)
            
    elif method == 'tsne':
        try:
            from sklearn.manifold import TSNE
            print("t-SNE (t-Distributed Stochastic Neighbor Embedding):")
            print("  - NON-LINEAR dimensionality reduction")
            print("  - Excellent for visualization, preserves local structure")
            print("  - SLOWER than UMAP, but often produces clearer clusters")
            print("  - Note: t-SNE distances don't preserve global structure\n")
            
            print("Fitting t-SNE (this may take several minutes for large datasets)...")
            print("  Using PCA initialization for faster convergence...")
            
            # Initialize with PCA for faster convergence
            pca_init = PCA(n_components=50, random_state=42)
            embeddings_pca = pca_init.fit_transform(embeddings)
            
            tsne = TSNE(
                n_components=3,
                random_state=42,
                perplexity=30,  # Good default for medium datasets
                n_iter=1000,
                init='pca',  # Use PCA initialization
                learning_rate='auto'
            )
            reduced = tsne.fit_transform(embeddings_pca)
            print("✓ t-SNE reduction complete")
            print("  t-SNE emphasizes local clusters - similar courses will be close together")
            
            return reduced, 't-SNE'
        except ImportError:
            print("⚠️  sklearn version may not support 3D t-SNE")
            print("Falling back to UMAP...\n")
            return reduce_to_3d(embeddings, method='umap', show_diagnostics=False)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'umap', or 'tsne'")


def create_3d_plot(df, x_col='x', y_col='y', z_col='z', color_by='department', output_path='visualizations/embedding_3d.html', method_name='PCA'):
    """
    Create an interactive 3D scatter plot using Plotly.
    
    Args:
        df: DataFrame with x, y, z coordinates and metadata
        x_col, y_col, z_col: Column names for 3D coordinates
        color_by: Column to use for coloring ('department' or 'graduate')
        output_path: Path to save HTML file
    """
    print(f"Creating 3D visualization (colored by {color_by})...")
    
    # Prepare color column
    if color_by == 'department':
        color_col = 'department'
        color_discrete_map = None  # Use default color scheme
    elif color_by == 'graduate':
        color_col = 'is_graduate_level'
        color_discrete_map = {True: '#FF6B6B', False: '#4ECDC4'}  # Red for grad, teal for undergrad
    else:
        raise ValueError(f"Unknown color_by option: {color_by}. Use 'department' or 'graduate'")
    
    # Create hover text with course info
    hover_data = []
    for idx, row in df.iterrows():
        hover_text = (
            f"<b>{row['course_id']}</b><br>"
            f"{row['title']}<br>"
            f"Dept: {row['department']}<br>"
            f"Level: {'Graduate' if row['is_graduate_level'] else 'Undergraduate'}<br>"
            f"<i>{row['description'][:100]}...</i>" if row['description'] else ""
        )
        hover_data.append(hover_text)
    
    df['hover_text'] = hover_data
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        color_discrete_map=color_discrete_map,
        hover_name='course_id',
        hover_data={'title': True, 'department': True, 'is_graduate_level': True},
        labels={
            x_col: f'{method_name} 1',
            y_col: f'{method_name} 2',
            z_col: f'{method_name} 3',
            color_col: 'Department' if color_by == 'department' else 'Level'
        },
        title=f'Course Embeddings in 3D Space ({method_name})',
        width=1200,
        height=800
    )
    
    # Customize hover template
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' +
                      f'{method_name}1: %{{x:.2f}}<br>' +
                      f'{method_name}2: %{{y:.2f}}<br>' +
                      f'{method_name}3: %{{z:.2f}}<extra></extra>',
        marker=dict(size=4, opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey'))
    )
    
    # Update layout for better viewing
    fig.update_layout(
        scene=dict(
            xaxis_title=f'{method_name} Dimension 1',
            yaxis_title=f'{method_name} Dimension 2',
            zaxis_title=f'{method_name} Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        font=dict(size=12),
        title_font_size=16
    )
    
    # Ensure output directory exists
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to HTML
        print(f"Saving interactive plot to {output_path}...")
        fig.write_html(output_path)
        print(f"✓ Saved! Open {output_path} in your browser to explore the visualization.")
    
    return fig


def save_image(fig, output_path, img_format='png', width=1200, height=800):
    """
    Save the plot as a static image file.
    
    Args:
        fig: Plotly figure object
        output_path: Base output path (will have extension added)
        img_format: Image format - 'png', 'pdf', 'svg', or 'jpg'
        width: Image width in pixels
        height: Image height in pixels
    """
    # Map format to file extension
    format_extensions = {
        'png': '.png',
        'pdf': '.pdf',
        'svg': '.svg',
        'jpg': '.jpg',
        'jpeg': '.jpg'
    }
    
    if img_format.lower() not in format_extensions:
        raise ValueError(f"Unsupported format: {img_format}. Use: {', '.join(format_extensions.keys())}")
    
    # Generate image output path
    base_path = os.path.splitext(output_path)[0]  # Remove .html if present
    img_path = base_path + format_extensions[img_format.lower()]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    
    print(f"Saving static image to {img_path}...")
    try:
        # Update figure size for image export
        fig.update_layout(width=width, height=height)
        
        # Export to image using kaleido
        fig.write_image(img_path, width=width, height=height, scale=2)  # scale=2 for high-res
        print(f"✓ Image saved! ({img_format.upper()}, {width}x{height}px)")
        return img_path
    except Exception as e:
        print(f"⚠️  Error saving image: {e}")
        print("   Make sure kaleido is installed: pip install kaleido")
        print("   Falling back to HTML export only.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Create 3D visualization of course embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/202508_processed.pkl',
        help='Path to processed pickle file (default: data/202508_processed.pkl)'
    )
    parser.add_argument(
        '--color-by',
        type=str,
        choices=['department', 'graduate'],
        default='department',
        help='Color scheme: department or graduate level (default: department)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='visualizations/embedding_3d.html',
        help='Output HTML file path (default: visualizations/embedding_3d.html)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['pca', 'umap', 'tsne'],
        default='umap',
        help='Dimensionality reduction method: pca (linear, fast), umap (non-linear, recommended), or tsne (non-linear, slower but clearer clusters). Default: umap'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of courses to visualize (default: None, show all)'
    )
    parser.add_argument(
        '--img',
        action='store_true',
        help='Also save as static image file (PNG by default)'
    )
    parser.add_argument(
        '--img-format',
        type=str,
        choices=['png', 'pdf', 'svg', 'jpg'],
        default='png',
        help='Image format when using --img flag (default: png)'
    )
    parser.add_argument(
        '--img-only',
        action='store_true',
        help='Save only as image, skip HTML file'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, args.data_path)
    output_path = os.path.join(project_root, args.output)
    
    # Load data
    df = load_data(data_path)
    
    # Limit courses if specified
    if args.limit and args.limit < len(df):
        print(f"Limiting to {args.limit} courses (random sample)...")
        df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = np.array(df['embedding'].tolist())
    print(f"Embedding shape: {embeddings.shape}")
    
    # Reduce to 3D
    reduced_3d, method_name = reduce_to_3d(embeddings, method=args.method)
    
    # Add 3D coordinates to dataframe
    df['x'] = reduced_3d[:, 0]
    df['y'] = reduced_3d[:, 1]
    df['z'] = reduced_3d[:, 2]
    
    # Create visualization
    # Skip HTML if img-only is set
    html_output_path = None if args.img_only else output_path
    fig = create_3d_plot(df, color_by=args.color_by, output_path=html_output_path, method_name=method_name)
    
    # Save image if requested
    if args.img or args.img_only:
        if args.img_only:
            print("\nSkipping HTML export (--img-only flag set)")
        save_image(
            fig, 
            output_path, 
            img_format=args.img_format
        )
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

