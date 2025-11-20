"""View embeddings from a BDH checkpoint.

This script loads a checkpoint and displays the embedding weights,
allowing you to inspect the learned token representations.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import bdh


def load_checkpoint_embeddings(checkpoint_path):
    """Load embeddings from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing embedding information
    """
    # Add BDHConfig to safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([bdh.BDHConfig])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Extract embeddings and related weights
    embed_weight = state_dict['embed.weight']  # Shape: (vocab_size, n_embd)
    lm_head = state_dict['lm_head']  # Shape: (n_embd, vocab_size)
    
    # Extract config if available
    config = checkpoint.get('config', None)
    
    return {
        'embed_weight': embed_weight,
        'lm_head': lm_head,
        'config': config,
        'step': checkpoint.get('step', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown'),
        'vocab_size': embed_weight.shape[0],
        'embedding_dim': embed_weight.shape[1],
    }


def print_embedding_stats(info):
    """Print statistics about the embeddings."""
    embed = info['embed_weight']
    lm_head = info['lm_head']
    
    print("=" * 70)
    print("CHECKPOINT INFORMATION")
    print("=" * 70)
    print(f"Checkpoint Step: {info['step']}")
    print(f"Loss: {info['loss']}")
    print(f"Vocabulary Size: {info['vocab_size']}")
    print(f"Embedding Dimension: {info['embedding_dim']}")
    
    if info['config'] is not None:
        print(f"\nModel Configuration:")
        print(f"  Layers: {info['config'].n_layer}")
        print(f"  Embedding Dim: {info['config'].n_embd}")
        print(f"  Heads: {info['config'].n_head}")
        print(f"  Dropout: {info['config'].dropout}")
    
    print("\n" + "=" * 70)
    print("EMBEDDING STATISTICS")
    print("=" * 70)
    
    print(f"\nInput Embeddings (embed.weight):")
    print(f"  Shape: {tuple(embed.shape)}")
    print(f"  Mean: {embed.mean().item():.6f}")
    print(f"  Std: {embed.std().item():.6f}")
    print(f"  Min: {embed.min().item():.6f}")
    print(f"  Max: {embed.max().item():.6f}")
    print(f"  Norm (Frobenius): {torch.norm(embed).item():.6f}")
    
    print(f"\nOutput Embeddings (lm_head):")
    print(f"  Shape: {tuple(lm_head.shape)}")
    print(f"  Mean: {lm_head.mean().item():.6f}")
    print(f"  Std: {lm_head.std().item():.6f}")
    print(f"  Min: {lm_head.min().item():.6f}")
    print(f"  Max: {lm_head.max().item():.6f}")
    print(f"  Norm (Frobenius): {torch.norm(lm_head).item():.6f}")


def view_specific_embeddings(info, token_ids, show_full=False):
    """View embeddings for specific token IDs.
    
    Args:
        info: Embedding info dictionary
        token_ids: List of token IDs to view
        show_full: Whether to show full embedding vectors
    """
    embed = info['embed_weight']
    
    print("\n" + "=" * 70)
    print("SPECIFIC TOKEN EMBEDDINGS")
    print("=" * 70)
    
    for token_id in token_ids:
        if token_id >= info['vocab_size']:
            print(f"\nToken {token_id}: OUT OF RANGE (vocab_size={info['vocab_size']})")
            continue
            
        vec = embed[token_id]
        print(f"\nToken {token_id}:")
        print(f"  Mean: {vec.mean().item():.6f}")
        print(f"  Std: {vec.std().item():.6f}")
        print(f"  L2 Norm: {torch.norm(vec).item():.6f}")
        
        if show_full:
            print(f"  Vector: {vec.numpy()}")
        else:
            # Show first and last few values
            vec_np = vec.numpy()
            if len(vec_np) > 10:
                print(f"  First 5: {vec_np[:5]}")
                print(f"  Last 5:  {vec_np[-5:]}")
            else:
                print(f"  Vector: {vec_np}")


def compute_similarity_matrix(info, token_ids=None, top_k=10, tokenizer=None):
    """Compute and display similarity between embeddings.
    
    Args:
        info: Embedding info dictionary
        token_ids: Optional list of token IDs to compare (default: random sample)
        top_k: Number of most similar tokens to show for each token
        tokenizer: Optional tokenizer to decode token IDs to strings
    """
    embed = info['embed_weight']
    
    if token_ids is None:
        # Sample random tokens if not specified
        n_samples = min(10, info['vocab_size'])
        token_ids = torch.randint(0, info['vocab_size'], (n_samples,)).tolist()
    
    print("\n" + "=" * 70)
    print("EMBEDDING SIMILARITIES (Cosine)")
    print("=" * 70)
    
    # Normalize embeddings for cosine similarity
    embed_norm = torch.nn.functional.normalize(embed, p=2, dim=1)
    
    for token_id in token_ids:
        if token_id >= info['vocab_size']:
            continue
        
        # Decode the query token if tokenizer available
        token_str = ""
        if tokenizer:
            try:
                decoded = tokenizer.decode([token_id])
                token_str = f" ({repr(decoded)})"
            except:
                pass
            
        # Compute cosine similarity with all tokens
        vec = embed_norm[token_id:token_id+1]
        similarities = (embed_norm @ vec.T).squeeze()
        
        # Get top-k most similar (excluding itself)
        top_similarities, top_indices = torch.topk(similarities, k=min(top_k+1, info['vocab_size']))
        
        print(f"\nToken {token_id}{token_str} - Top {top_k} most similar tokens:")
        for i, (idx, sim) in enumerate(zip(top_indices.tolist(), top_similarities.tolist())):
            if idx == token_id:
                continue
            
            # Decode similar token if tokenizer available
            similar_str = ""
            if tokenizer:
                try:
                    decoded = tokenizer.decode([idx])
                    similar_str = f" {repr(decoded)}"
                except:
                    pass
            
            print(f"  {i+1}. Token {idx:5d}{similar_str}: {sim:.6f}")
            if i >= top_k - 1:
                break


def export_embeddings(info, output_path, format='npy'):
    """Export embeddings to file.
    
    Args:
        info: Embedding info dictionary
        output_path: Output file path (without extension)
        format: Export format ('npy', 'txt', 'csv')
    """
    embed = info['embed_weight'].numpy()
    lm_head = info['lm_head'].numpy()
    
    if format == 'npy':
        np.save(f"{output_path}_embed.npy", embed)
        np.save(f"{output_path}_lmhead.npy", lm_head)
        print(f"\nExported embeddings to:")
        print(f"  {output_path}_embed.npy")
        print(f"  {output_path}_lmhead.npy")
        
    elif format == 'txt':
        np.savetxt(f"{output_path}_embed.txt", embed, fmt='%.6f')
        np.savetxt(f"{output_path}_lmhead.txt", lm_head, fmt='%.6f')
        print(f"\nExported embeddings to:")
        print(f"  {output_path}_embed.txt")
        print(f"  {output_path}_lmhead.txt")
        
    elif format == 'csv':
        np.savetxt(f"{output_path}_embed.csv", embed, delimiter=',', fmt='%.6f')
        np.savetxt(f"{output_path}_lmhead.csv", lm_head, delimiter=',', fmt='%.6f')
        print(f"\nExported embeddings to:")
        print(f"  {output_path}_embed.csv")
        print(f"  {output_path}_lmhead.csv")


def visualize_embedding_heatmap(info, token_ids=None, save_path=None):
    """Create a heatmap visualization of embeddings.
    
    Args:
        info: Embedding info dictionary
        token_ids: Optional list of token IDs to visualize (default: first 50)
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("\nMatplotlib and seaborn required for visualization.")
        print("Install with: pip install matplotlib seaborn")
        return
    
    embed = info['embed_weight']
    
    if token_ids is None:
        # Show first N tokens
        n_tokens = min(50, info['vocab_size'])
        token_ids = list(range(n_tokens))
    
    # Extract embeddings for selected tokens
    selected_embeds = embed[token_ids].numpy()
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(selected_embeds, cmap='coolwarm', center=0, 
                xticklabels=False, yticklabels=token_ids,
                cbar_kws={'label': 'Embedding Value'})
    plt.title(f'Embedding Heatmap (Step {info["step"]})')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Token ID')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nHeatmap saved to: {save_path}")
    else:
        plt.show()


def visualize_cluster_map(info, n_tokens=None, method='tsne', save_path=None, tokenizer=None, interactive=True):
    """Create a 2D visualization showing clusters of similar tokens.
    
    Args:
        info: Embedding info dictionary
        n_tokens: Number of tokens to visualize (default: min(500, vocab_size))
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        save_path: Optional path to save the figure
        tokenizer: Optional tokenizer to label points with decoded tokens
        interactive: If True, create interactive plot with hover tooltips (requires plotly)
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError:
        print("\nMatplotlib and scikit-learn required for visualization.")
        print("Install with: pip install matplotlib scikit-learn")
        return
    
    embed = info['embed_weight'].numpy()
    
    # Determine number of tokens to visualize
    if n_tokens is None:
        n_tokens = min(500, info['vocab_size'])
    n_tokens = min(n_tokens, info['vocab_size'])
    
    # Sample or use first n tokens
    if n_tokens < info['vocab_size']:
        # Sample uniformly across vocabulary
        token_indices = np.linspace(0, info['vocab_size']-1, n_tokens, dtype=int)
    else:
        token_indices = np.arange(n_tokens)
    
    embeddings_subset = embed[token_indices]
    
    print(f"\nComputing {method.upper()} projection for {n_tokens} tokens...")
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_tokens-1))
        coords_2d = reducer.fit_transform(embeddings_subset)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings_subset)
        print(f"Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
    elif method.lower() == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(embeddings_subset)
        except ImportError:
            print("UMAP not available. Install with: pip install umap-learn")
            print("Falling back to t-SNE...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_tokens-1))
            coords_2d = reducer.fit_transform(embeddings_subset)
            method = 'tsne'
    else:
        print(f"Unknown method '{method}'. Using t-SNE.")
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_tokens-1))
        coords_2d = reducer.fit_transform(embeddings_subset)
        method = 'tsne'
    
    # Prepare token labels for hover text
    token_labels = []
    for token_id in token_indices:
        if tokenizer:
            try:
                decoded = tokenizer.decode([token_id])
                # Create readable label
                if decoded.isprintable() and decoded.strip():
                    token_labels.append(f"{repr(decoded)} (ID: {token_id})")
                else:
                    token_labels.append(f"<byte 0x{token_id:02x}> (ID: {token_id})")
            except:
                token_labels.append(f"Token ID: {token_id}")
        else:
            token_labels.append(f"Token ID: {token_id}")
    
    # Create visualization
    if interactive and save_path and not save_path.endswith('.png'):
        # Use plotly for interactive visualization
        try:
            import plotly.graph_objects as go
            
            # Create base scatter plot
            fig = go.Figure(data=go.Scatter(
                x=coords_2d[:, 0],
                y=coords_2d[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=token_indices,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Token ID"),
                    line=dict(width=0.5, color='black')
                ),
                text=token_labels,
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Token Embedding Cluster Map - {method.upper()} (Step {info["step"]})<br>'
                      f'{n_tokens} tokens from vocabulary of {info["vocab_size"]}',
                xaxis_title=f'{method.upper()} Dimension 1',
                yaxis_title=f'{method.upper()} Dimension 2',
                width=1200,
                height=900,
                hovermode='closest'
            )
            
            if save_path.endswith('.html'):
                # First write the basic HTML
                fig.write_html(save_path)
                
                # Read it back and inject custom search functionality
                with open(save_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Inject custom CSS and search UI before </head>
                custom_head = """
    <style>
        body {
            padding-top: 80px;
        }
        #search-container {
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: white;
            padding: 15px;
            border: 2px solid #333;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        #search-input {
            padding: 8px;
            width: 200px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
        }
        #search-btn, #clear-btn {
            padding: 8px 15px;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        #search-btn {
            background: #007bff;
        }
        #search-btn:hover {
            background: #0056b3;
        }
        #clear-btn {
            background: #6c757d;
        }
        #clear-btn:hover {
            background: #545b62;
        }
        #result-info {
            font-size: 12px;
            color: #666;
            white-space: nowrap;
        }
    </style>
</head>"""
                html_content = html_content.replace('</head>', custom_head)
                
                # Inject search UI after <body>
                search_ui = """<body>
    <div id="search-container">
        <input type="text" id="search-input" placeholder="Search for token...">
        <button id="search-btn">Find</button>
        <button id="clear-btn">Clear</button>
        <div id="result-info"></div>
    </div>
"""
                html_content = html_content.replace('<body>', search_ui)
                
                # Inject search JavaScript before </body>
                search_script = """
    <script>
        document.getElementById('search-btn').addEventListener('click', searchToken);
        document.getElementById('clear-btn').addEventListener('click', clearHighlight);
        document.getElementById('search-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') searchToken();
        });
        
        function searchToken() {
            const searchTerm = document.getElementById('search-input').value.trim();
            if (!searchTerm) return;
            
            const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            
            // Use _fullData which has the actual coordinate arrays
            const fullTrace = plotDiv._fullData[0];
            const labels = fullTrace.text;
            const matches = [];
            
            labels.forEach((label, idx) => {
                const tokenMatch = label.match(/^['"](.+?)['"]|^<byte 0x[0-9a-f]+>/i);
                if (tokenMatch) {
                    const token = tokenMatch[1] || tokenMatch[0];
                    if (token.toLowerCase().includes(searchTerm.toLowerCase())) {
                        matches.push(idx);
                    }
                }
            });
            
            if (matches.length === 0) {
                document.getElementById('result-info').textContent = `No matches found for "${searchTerm}"`;
                return;
            }
            
            // Remove old highlight traces
            while (plotDiv.data.length > 1) {
                Plotly.deleteTraces(plotDiv, 1);
            }
            
            // Build highlight data using _fullData
            const highlightX = [];
            const highlightY = [];
            const highlightText = [];
            
            matches.forEach(idx => {
                highlightX.push(fullTrace.x[idx]);
                highlightY.push(fullTrace.y[idx]);
                highlightText.push(labels[idx]);
            });
            
            const highlightTrace = {
                x: highlightX,
                y: highlightY,
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: 16,
                    color: 'rgba(255, 0, 0, 0.6)',
                    symbol: 'circle',
                    line: {
                        color: 'red',
                        width: 2
                    }
                },
                text: highlightText,
                hovertemplate: '<b>MATCH</b><br>%{text}<extra></extra>',
                showlegend: false
            };
            
            Plotly.addTraces(plotDiv, highlightTrace);
            
            document.getElementById('result-info').textContent = 
                `Found ${matches.length} match${matches.length > 1 ? 'es' : ''} for "${searchTerm}"`;
        }
        
        function clearHighlight() {
            const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
            
            // Remove highlight trace if it exists
            if (plotDiv.data.length > 1) {
                Plotly.deleteTraces(plotDiv, 1);
            }
            
            document.getElementById('search-input').value = '';
            document.getElementById('result-info').textContent = '';
        }
    </script>
</body>"""
                html_content = html_content.replace('</body>', search_script)
                
                # Write the modified HTML back
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
                print(f"\nInteractive cluster map saved to: {save_path}")
                print("Use the search box at the top to find and highlight tokens")
            else:
                fig.show()
        except ImportError:
            print("\nPlotly not available for interactive plots. Install with: pip install plotly")
            print("Falling back to static matplotlib visualization...")
            interactive = False
    
    if not interactive or (save_path and save_path.endswith('.png')):
        # Use matplotlib for static visualization
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(16, 12))
        
        # Color by token ID
        scatter = plt.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                             c=token_indices, cmap='viridis', 
                             alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add labels for a subset of tokens (keep sparse to avoid crowding)
        if tokenizer and n_tokens <= 100:
            for i, token_id in enumerate(token_indices):
                try:
                    decoded = tokenizer.decode([token_id])
                    # Only label printable tokens
                    if decoded and decoded.isprintable() and len(decoded.strip()) > 0:
                        plt.annotate(repr(decoded), 
                                   (coords_2d[i, 0], coords_2d[i, 1]),
                                   fontsize=8, alpha=0.7,
                                   xytext=(5, 5), textcoords='offset points')
                except:
                    pass
        
        plt.colorbar(scatter, label='Token ID')
        plt.title(f'Token Embedding Cluster Map - {method.upper()} (Step {info["step"]})\n'
                 f'{n_tokens} tokens from vocabulary of {info["vocab_size"]}')
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            # Convert .html extension to .png for matplotlib fallback
            if save_path.endswith('.html'):
                save_path = save_path.replace('.html', '.png')
                print(f"\nNote: Saving as PNG instead of HTML (plotly not available)")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nCluster map saved to: {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="View embeddings from BDH checkpoint")
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--tokens',
        type=str,
        nargs='+',
        help='Specific tokens to view (e.g., --tokens A B C "hello")'
    )
    parser.add_argument(
        '--token-ids',
        type=int,
        nargs='+',
        help='Specific token IDs to view (e.g., --token-ids 0 1 2 255)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Show full embedding vectors (not just summary stats)'
    )
    parser.add_argument(
        '--similarity',
        action='store_true',
        help='Show similarity matrix for tokens'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of most similar tokens to show (default: 10)'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export embeddings to file (specify output path without extension)'
    )
    parser.add_argument(
        '--format',
        choices=['npy', 'txt', 'csv'],
        default='npy',
        help='Export format (default: npy)'
    )
    parser.add_argument(
        '--heatmap',
        action='store_true',
        help='Generate embedding heatmap visualization'
    )
    parser.add_argument(
        '--heatmap-tokens',
        type=int,
        nargs='+',
        help='Specific token IDs for heatmap (default: first 50)'
    )
    parser.add_argument(
        '--save-heatmap',
        type=str,
        help='Save heatmap to file instead of displaying'
    )
    parser.add_argument(
        '--cluster-map',
        action='store_true',
        help='Generate 2D cluster map showing groups of similar tokens'
    )
    parser.add_argument(
        '--cluster-method',
        choices=['tsne', 'pca', 'umap'],
        default='tsne',
        help='Dimensionality reduction method for cluster map (default: tsne)'
    )
    parser.add_argument(
        '--cluster-tokens',
        type=int,
        help='Number of tokens to include in cluster map (default: 500)'
    )
    parser.add_argument(
        '--save-cluster',
        type=str,
        help='Save cluster map to file (.html for interactive, .png for static)'
    )
    
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading checkpoint: {args.checkpoint}")
    info = load_checkpoint_embeddings(args.checkpoint)
    
    # Load tokenizer if available from checkpoint config
    tokenizer = None
    if info['config'] is not None:
        try:
            # Try to get tokenizer config from checkpoint
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            tokenizer_config = checkpoint.get('tokenizer_config', {})
            if tokenizer_config:
                from tokenizers import create_tokenizer
                tokenizer = create_tokenizer(
                    tokenizer_config.get('type', 'byte'),
                    tokenizer_config.get('name', None)
                )
        except Exception as e:
            # Fallback to byte tokenizer if error
            try:
                from tokenizers import ByteTokenizer
                tokenizer = ByteTokenizer()
            except:
                pass
    
    # Print statistics
    print_embedding_stats(info)
    
    # Convert token strings to IDs if needed
    token_ids = None
    if args.tokens:
        if tokenizer is None:
            print("\nWarning: Cannot decode tokens without a tokenizer. Use --token-ids instead.")
        else:
            token_ids = []
            for token_str in args.tokens:
                # Encode the token string and get the first token ID
                encoded = tokenizer.encode(token_str)
                if encoded:
                    token_ids.append(encoded[0])
                    print(f"Token '{token_str}' -> ID {encoded[0]}")
                else:
                    print(f"Warning: Could not encode token '{token_str}'")
    elif args.token_ids:
        token_ids = args.token_ids
    
    # View specific embeddings
    if token_ids:
        view_specific_embeddings(info, token_ids, show_full=args.full)
    
    # Show similarity matrix
    if args.similarity:
        compute_similarity_matrix(info, token_ids, top_k=args.top_k, tokenizer=tokenizer)
    
    # Export embeddings
    if args.export:
        export_embeddings(info, args.export, format=args.format)
    
    # Generate heatmap
    if args.heatmap:
        visualize_embedding_heatmap(info, args.heatmap_tokens, save_path=args.save_heatmap)
    
    # Generate cluster map
    if args.cluster_map:
        visualize_cluster_map(info, n_tokens=args.cluster_tokens, 
                             method=args.cluster_method, 
                             save_path=args.save_cluster,
                             tokenizer=tokenizer)


if __name__ == "__main__":
    main()
