"""Utility functions for trace visualization and manipulation."""

from typing import Optional, Union, List, Tuple
import numpy as np
import torch


def _to_numpy(trace: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert trace to numpy array."""
    if isinstance(trace, torch.Tensor):
        return trace.detach().cpu().numpy()
    return np.asarray(trace)


def _is_deterministic(trace: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if a trace is deterministic (one-hot encoded).
    
    A trace is considered deterministic if each row has exactly one value
    close to 1 and the rest close to 0.
    """
    max_vals = trace.max(axis=1)
    sum_vals = trace.sum(axis=1)
    return np.allclose(max_vals, 1.0, atol=tol) and np.allclose(sum_vals, 1.0, atol=tol)


def visualize_trace(
    trace: Union[torch.Tensor, np.ndarray],
    activity_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
    cmap: str = "Blues",
    show_colorbar: bool = True,
    show_values: bool = False,
    value_fmt: str = ".2f",
    ax=None,
):
    """
    Visualize a trace as a heatmap.
    
    Works for both deterministic (one-hot) and stochastic (probability) traces.
    
    Args:
        trace: Tensor of shape (L, C) where L is sequence length and C is number of classes.
               For deterministic traces, each row is one-hot encoded.
               For stochastic traces, each row is a probability distribution.
        activity_names: Optional list of activity/class names for y-axis labels.
        title: Optional title for the plot. If None, auto-generates based on trace type.
        figsize: Figure size as (width, height).
        cmap: Colormap name for the heatmap.
        show_colorbar: Whether to display the colorbar.
        show_values: Whether to annotate cells with their values.
        value_fmt: Format string for value annotations.
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects.
    """
    import matplotlib.pyplot as plt
    
    trace_np = _to_numpy(trace)
    
    if trace_np.ndim != 2:
        raise ValueError(f"Expected trace of shape (L, C), got shape {trace_np.shape}")
    
    seq_len, num_classes = trace_np.shape
    is_det = _is_deterministic(trace_np)
    
    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Transpose for visualization: classes on y-axis, time on x-axis
    trace_display = trace_np.T
    
    # Create heatmap
    im = ax.imshow(trace_display, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    
    # Set labels
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Activity")
    
    # Set title
    if title is None:
        trace_type = "Deterministic" if is_det else "Stochastic"
        title = f"{trace_type} Trace Visualization"
    ax.set_title(title)
    
    # Set y-axis labels (activities)
    if activity_names is not None:
        if len(activity_names) != num_classes:
            raise ValueError(
                f"Number of activity names ({len(activity_names)}) "
                f"doesn't match number of classes ({num_classes})"
            )
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(activity_names)
    else:
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels([f"Class {i}" for i in range(num_classes)])
    
    # Set x-axis ticks
    if seq_len <= 20:
        ax.set_xticks(range(seq_len))
    else:
        # Show fewer ticks for longer sequences
        tick_step = max(1, seq_len // 10)
        ax.set_xticks(range(0, seq_len, tick_step))
    
    # Add colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, label="Probability")
        if is_det:
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["0", "1"])
    
    # Annotate cells with values
    if show_values:
        for i in range(num_classes):
            for j in range(seq_len):
                val = trace_display[i, j]
                # Choose text color based on background
                text_color = "white" if val > 0.5 else "black"
                ax.text(
                    j, i, format(val, value_fmt),
                    ha="center", va="center", color=text_color, fontsize=8
                )
    
    plt.tight_layout()
    return fig, ax


def visualize_trace_interactive(
    trace: Union[torch.Tensor, np.ndarray],
    activity_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 500,
    colorscale: str = "Blues",
):
    """
    Create an interactive trace visualization using Plotly.
    
    Provides hover information, zoom, and pan capabilities.
    
    Args:
        trace: Tensor of shape (L, C) where L is sequence length and C is number of classes.
               For deterministic traces, each row is one-hot encoded.
               For stochastic traces, each row is a probability distribution.
        activity_names: Optional list of activity/class names for y-axis labels.
        title: Optional title for the plot. If None, auto-generates based on trace type.
        width: Width of the figure in pixels.
        height: Height of the figure in pixels.
        colorscale: Plotly colorscale name for the heatmap.
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure.
    """
    import plotly.graph_objects as go
    
    trace_np = _to_numpy(trace)
    
    if trace_np.ndim != 2:
        raise ValueError(f"Expected trace of shape (L, C), got shape {trace_np.shape}")
    
    seq_len, num_classes = trace_np.shape
    is_det = _is_deterministic(trace_np)
    
    # Transpose for visualization: classes on y-axis, time on x-axis
    trace_display = trace_np.T
    
    # Generate labels
    if activity_names is None:
        activity_names = [f"Class {i}" for i in range(num_classes)]
    elif len(activity_names) != num_classes:
        raise ValueError(
            f"Number of activity names ({len(activity_names)}) "
            f"doesn't match number of classes ({num_classes})"
        )
    
    time_labels = [str(i) for i in range(seq_len)]
    
    # Create custom hover text
    hover_text = []
    for i, activity in enumerate(activity_names):
        row_text = []
        for j in range(seq_len):
            prob = trace_display[i, j]
            row_text.append(
                f"Time: {j}<br>"
                f"Activity: {activity}<br>"
                f"Probability: {prob:.4f}"
            )
        hover_text.append(row_text)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=trace_display,
        x=time_labels,
        y=activity_names,
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        hoverinfo="text",
        text=hover_text,
        colorbar=dict(
            title="Probability",
            tickvals=[0, 0.25, 0.5, 0.75, 1] if not is_det else [0, 1],
            ticktext=["0", "0.25", "0.5", "0.75", "1"] if not is_det else ["0", "1"],
        ),
    ))
    
    # Set title
    if title is None:
        trace_type = "Deterministic" if is_det else "Stochastic"
        title = f"{trace_type} Trace Visualization"
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(
            title="Time Step",
            tickmode="linear" if seq_len <= 20 else "auto",
            dtick=1 if seq_len <= 20 else None,
        ),
        yaxis=dict(
            title="Activity",
            tickmode="array",
            tickvals=list(range(num_classes)),
            ticktext=activity_names,
        ),
        width=width,
        height=height,
    )
    
    return fig


def visualize_traces_comparison(
    traces: List[Union[torch.Tensor, np.ndarray]],
    labels: Optional[List[str]] = None,
    activity_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    cmap: str = "Blues",
):
    """
    Visualize multiple traces side by side for comparison.
    
    Useful for comparing original vs. denoised traces, or ground truth vs. prediction.
    
    Args:
        traces: List of traces, each of shape (L, C).
        labels: Optional list of labels for each trace (e.g., ["Original", "Denoised"]).
        activity_names: Optional list of activity/class names for y-axis labels.
        title: Optional overall title for the figure.
        figsize: Figure size. If None, auto-calculated based on number of traces.
        cmap: Colormap name for the heatmaps.
        
    Returns:
        tuple: (fig, axes) matplotlib figure and array of axes objects.
    """
    import matplotlib.pyplot as plt
    
    n_traces = len(traces)
    
    if labels is None:
        labels = [f"Trace {i+1}" for i in range(n_traces)]
    elif len(labels) != n_traces:
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match number of traces ({n_traces})"
        )
    
    if figsize is None:
        figsize = (5 * n_traces, 5)
    
    fig, axes = plt.subplots(1, n_traces, figsize=figsize)
    
    if n_traces == 1:
        axes = [axes]
    
    for i, (trace, label) in enumerate(zip(traces, labels)):
        visualize_trace(
            trace,
            activity_names=activity_names,
            title=label,
            cmap=cmap,
            show_colorbar=(i == n_traces - 1),  # Only show colorbar on last plot
            ax=axes[i],
        )
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    return fig, axes


def visualize_traces_comparison_interactive(
    traces: List[Union[torch.Tensor, np.ndarray]],
    labels: Optional[List[str]] = None,
    activity_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    width: int = 400,
    height: int = 400,
    colorscale: str = "Blues",
):
    """
    Create an interactive comparison of multiple traces using Plotly subplots.
    
    Args:
        traces: List of traces, each of shape (L, C).
        labels: Optional list of labels for each trace.
        activity_names: Optional list of activity/class names for y-axis labels.
        title: Optional overall title for the figure.
        width: Width per subplot in pixels.
        height: Height of the figure in pixels.
        colorscale: Plotly colorscale name for the heatmaps.
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure with subplots.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    n_traces = len(traces)
    
    if labels is None:
        labels = [f"Trace {i+1}" for i in range(n_traces)]
    elif len(labels) != n_traces:
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match number of traces ({n_traces})"
        )
    
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=n_traces,
        subplot_titles=labels,
        horizontal_spacing=0.1,
    )
    
    for i, trace in enumerate(traces):
        trace_np = _to_numpy(trace)
        seq_len, num_classes = trace_np.shape
        trace_display = trace_np.T
        
        # Generate labels
        if activity_names is None:
            y_labels = [f"Class {j}" for j in range(num_classes)]
        else:
            y_labels = activity_names
        
        time_labels = [str(j) for j in range(seq_len)]
        
        # Create hover text
        hover_text = []
        for j, activity in enumerate(y_labels):
            row_text = []
            for k in range(seq_len):
                prob = trace_display[j, k]
                row_text.append(
                    f"Time: {k}<br>"
                    f"Activity: {activity}<br>"
                    f"Probability: {prob:.4f}"
                )
            hover_text.append(row_text)
        
        fig.add_trace(
            go.Heatmap(
                z=trace_display,
                x=time_labels,
                y=y_labels,
                colorscale=colorscale,
                zmin=0,
                zmax=1,
                hoverinfo="text",
                text=hover_text,
                showscale=(i == n_traces - 1),  # Only show colorbar on last plot
            ),
            row=1,
            col=i + 1,
        )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
        width=width * n_traces,
        height=height,
    )
    
    # Update axes labels
    for i in range(n_traces):
        fig.update_xaxes(title_text="Time Step", row=1, col=i + 1)
        if i == 0:
            fig.update_yaxes(title_text="Activity", row=1, col=i + 1)
    
    return fig


def _get_color_palette(
    num_classes: int,
    cmap: Optional[str] = None,
    colors: Optional[List] = None,
) -> np.ndarray:
    """
    Get a color palette for the given number of classes.
    
    Args:
        num_classes: Number of classes to generate colors for.
        cmap: Optional matplotlib colormap name. If None, uses 'tab20' for <=20 classes,
              otherwise uses 'hsv'.
        colors: Optional list of colors to use directly. Overrides cmap if provided.
        
    Returns:
        np.ndarray: Array of shape (num_classes, 4) with RGBA colors.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    if colors is not None:
        # Convert provided colors to RGBA
        rgba_colors = [mcolors.to_rgba(c) for c in colors]
        if len(rgba_colors) < num_classes:
            raise ValueError(
                f"Not enough colors provided ({len(rgba_colors)}) for {num_classes} classes"
            )
        return np.array(rgba_colors[:num_classes])
    
    # Choose appropriate colormap
    if cmap is None:
        if num_classes <= 10:
            cmap = "tab10"
        elif num_classes <= 20:
            cmap = "tab20"
        else:
            cmap = "hsv"
    
    colormap = plt.get_cmap(cmap)
    
    if cmap in ["tab10", "tab20"]:
        # Discrete colormaps - sample directly
        colors_arr = [colormap(i) for i in range(num_classes)]
    else:
        # Continuous colormaps - sample evenly
        colors_arr = [colormap(i / num_classes) for i in range(num_classes)]
    
    return np.array(colors_arr)


def _trace_to_class_indices(trace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert trace to class indices (argmax at each time step).
    
    Works for both deterministic and stochastic traces:
    - Deterministic (one-hot): returns the index of the 1
    - Stochastic (probability): returns the index of max probability
    
    Args:
        trace: Array of shape (L, C).
        
    Returns:
        tuple: (class_indices, confidences) where:
            - class_indices: Array of shape (L,) with class indices
            - confidences: Array of shape (L,) with max probability at each step
                          (always 1.0 for deterministic traces)
    """
    class_indices = np.argmax(trace, axis=1)
    confidences = np.max(trace, axis=1)
    return class_indices, confidences


def visualize_trace_segmentation(
    trace: Union[torch.Tensor, np.ndarray],
    activity_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 1.5),
    cmap: Optional[str] = None,
    colors: Optional[List] = None,
    show_legend: bool = True,
    show_boundaries: bool = False,
    show_confidence: bool = False,
    ax=None,
) -> Tuple:
    """
    Visualize a trace as a colored segmentation bar.
    
    Each time step is shown as a colored segment where the color represents
    the activity class. Works for both deterministic and stochastic traces:
    - Deterministic traces: displays the one-hot class directly
    - Stochastic traces: displays the argmax (most probable) class
    
    Args:
        trace: Tensor of shape (L, C) where L is sequence length and C is number of classes.
               Supports both one-hot (deterministic) and probability (stochastic) traces.
        activity_names: Optional list of activity/class names for legend.
        title: Optional title for the plot. If None and trace is stochastic,
               auto-generates title indicating argmax was used.
        figsize: Figure size as (width, height).
        cmap: Colormap name for generating colors. If None, auto-selects based on num_classes.
        colors: Optional list of colors to use. Overrides cmap if provided.
        show_legend: Whether to display the legend.
        show_boundaries: Whether to show vertical lines at class boundaries.
        show_confidence: Whether to modulate color alpha by confidence (for stochastic traces).
                        Has no visual effect on deterministic traces.
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    trace_np = _to_numpy(trace)
    
    if trace_np.ndim != 2:
        raise ValueError(f"Expected trace of shape (L, C), got shape {trace_np.shape}")
    
    seq_len, num_classes = trace_np.shape
    is_det = _is_deterministic(trace_np)
    class_indices, confidences = _trace_to_class_indices(trace_np)
    
    # Get color palette
    palette = _get_color_palette(num_classes, cmap=cmap, colors=colors)
    
    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Create image array for the segmentation
    seg_image = np.zeros((1, seq_len, 4))  # (height=1, width=seq_len, RGBA)
    for t in range(seq_len):
        seg_image[0, t, :] = palette[class_indices[t]].copy()
        # Modulate alpha by confidence for stochastic traces
        if show_confidence and not is_det:
            # Scale alpha between 0.3 and 1.0 based on confidence
            seg_image[0, t, 3] = 0.3 + 0.7 * confidences[t]
    
    # Display segmentation
    ax.imshow(seg_image, aspect="auto", interpolation="nearest")
    
    # Show boundaries between different classes
    if show_boundaries:
        for t in range(1, seq_len):
            if class_indices[t] != class_indices[t - 1]:
                ax.axvline(x=t - 0.5, color="white", linewidth=0.5, alpha=0.7)
    
    # Remove y-axis ticks (single row)
    ax.set_yticks([])
    
    # Set x-axis
    ax.set_xlabel("Time Step")
    if seq_len <= 20:
        ax.set_xticks(range(seq_len))
    else:
        tick_step = max(1, seq_len // 10)
        ax.set_xticks(range(0, seq_len, tick_step))
    
    # Set title (indicate if stochastic trace was converted)
    if title is not None:
        ax.set_title(title)
    elif not is_det:
        avg_conf = np.mean(confidences)
        ax.set_title(f"Stochastic Trace (argmax, avg conf: {avg_conf:.2f})")
    
    # Create legend
    if show_legend:
        if activity_names is None:
            activity_names = [f"Class {i}" for i in range(num_classes)]
        
        # Only show legend for classes that appear in the trace
        unique_classes = np.unique(class_indices)
        patches = [
            mpatches.Patch(color=palette[i], label=activity_names[i])
            for i in unique_classes
        ]
        ax.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
        )
    
    plt.tight_layout()
    return fig, ax


def visualize_traces_segmentation(
    traces: List[Union[torch.Tensor, np.ndarray]],
    labels: Optional[List[str]] = None,
    activity_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Optional[tuple] = (12, 1.5),
    row_height: float = 0.8,
    cmap: Optional[str] = None,
    colors: Optional[List] = None,
    show_legend: bool = True,
    show_boundaries: bool = False,
    show_confidence: bool = False,
    vertical_gap: float = 0.1,
) -> Tuple:
    """
    Visualize multiple traces as stacked colored segmentation bars.
    
    Similar to video segmentation visualizations, each trace is shown as a
    horizontal bar with colored segments representing activities. Traces are
    stacked vertically for comparison (e.g., showing denoising progression).
    
    Works for both deterministic and stochastic traces:
    - Deterministic traces: displays the one-hot class directly
    - Stochastic traces: displays the argmax (most probable) class
    
    Args:
        traces: List of traces, each of shape (L, C). Can mix deterministic and stochastic.
        labels: Optional list of labels for each trace (e.g., ["Step 0", "Step 100"]).
        activity_names: Optional list of activity/class names for legend.
        title: Optional overall title for the figure.
        figsize: Figure size. If None, auto-calculated based on number of traces.
        row_height: Height of each trace row in inches.
        cmap: Colormap name for generating colors.
        colors: Optional list of colors to use. Overrides cmap if provided.
        show_legend: Whether to display the legend.
        show_boundaries: Whether to show vertical lines at class boundaries.
        show_confidence: Whether to modulate color alpha by confidence (for stochastic traces).
        vertical_gap: Vertical spacing between traces (0-1, as fraction of row height).
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    n_traces = len(traces)
    
    if labels is None:
        labels = [f"Trace {i+1}" for i in range(n_traces)]
    elif len(labels) != n_traces:
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match number of traces ({n_traces})"
        )
    
    # Convert all traces and get dimensions
    traces_np = [_to_numpy(t) for t in traces]
    seq_len = traces_np[0].shape[0]
    num_classes = traces_np[0].shape[1]
    
    # Check which traces are deterministic vs stochastic
    traces_is_det = [_is_deterministic(t) for t in traces_np]
    
    # Get color palette (shared across all traces)
    palette = _get_color_palette(num_classes, cmap=cmap, colors=colors)
    
    # Calculate figure size
    if figsize is None:
        width = max(10, seq_len / 20)
        height = n_traces * row_height + 1  # Extra space for legend
        figsize = (width, height)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Track unique classes for legend
    all_unique_classes = set()
    
    # Plot each trace as a row
    for i, (trace_np, label, is_det) in enumerate(zip(traces_np, labels, traces_is_det)):
        class_indices, confidences = _trace_to_class_indices(trace_np)
        all_unique_classes.update(class_indices)
        
        # Create image array for this trace
        seg_image = np.zeros((1, seq_len, 4))
        for t in range(seq_len):
            seg_image[0, t, :] = palette[class_indices[t]].copy()
            # Modulate alpha by confidence for stochastic traces
            if show_confidence and not is_det:
                seg_image[0, t, 3] = 0.3 + 0.7 * confidences[t]
        
        # Calculate y position (stack from top to bottom)
        y_pos = n_traces - 1 - i
        
        # Display segmentation as a bar
        extent = [0, seq_len, y_pos - 0.5 + vertical_gap/2, y_pos + 0.5 - vertical_gap/2]
        ax.imshow(seg_image, aspect="auto", extent=extent, interpolation="nearest")
        
        # Show boundaries between different classes
        if show_boundaries:
            for t in range(1, seq_len):
                if class_indices[t] != class_indices[t - 1]:
                    ax.vlines(
                        x=t,
                        ymin=y_pos - 0.5 + vertical_gap/2,
                        ymax=y_pos + 0.5 - vertical_gap/2,
                        color="white",
                        linewidth=0.5,
                        alpha=0.7,
                    )
    
    # Set y-axis with trace labels
    ax.set_yticks(range(n_traces))
    ax.set_yticklabels(labels[::-1])  # Reverse to match top-to-bottom order
    ax.set_ylim(-0.5, n_traces - 0.5)
    
    # Set x-axis
    ax.set_xlabel("Time Step")
    ax.set_xlim(0, seq_len)
    if seq_len <= 20:
        ax.set_xticks(range(seq_len + 1))
    else:
        tick_step = max(1, seq_len // 10)
        ax.set_xticks(range(0, seq_len + 1, tick_step))
    
    # Set title
    if title is not None:
        ax.set_title(title, fontsize=12)
    
    # Create legend
    if show_legend:
        if activity_names is None:
            activity_names = [f"Class {i}" for i in range(num_classes)]
        
        # Only show legend for classes that appear
        patches = [
            mpatches.Patch(color=palette[i], label=activity_names[i])
            for i in sorted(all_unique_classes)
        ]
        ax.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            title="Activities",
        )
    
    plt.tight_layout()
    return fig, ax


def visualize_traces_segmentation_interactive(
    traces: List[Union[torch.Tensor, np.ndarray]],
    labels: Optional[List[str]] = None,
    activity_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    width: int = 900,
    height: Optional[int] = None,
    row_height: int = 50,
    colors: Optional[List[str]] = None,
):
    """
    Create an interactive segmentation visualization using Plotly.
    
    Shows multiple traces as stacked colored bars with hover information.
    Works for both deterministic and stochastic traces:
    - Deterministic traces: displays the one-hot class directly
    - Stochastic traces: displays the argmax (most probable) class, with
                         average confidence shown in hover info
    
    Args:
        traces: List of traces, each of shape (L, C). Can mix deterministic and stochastic.
        labels: Optional list of labels for each trace.
        activity_names: Optional list of activity/class names.
        title: Optional title for the figure.
        width: Width of the figure in pixels.
        height: Height of the figure in pixels. If None, auto-calculated.
        row_height: Height per trace row in pixels.
        colors: Optional list of color strings for each class.
        
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    
    n_traces = len(traces)
    
    if labels is None:
        labels = [f"Trace {i+1}" for i in range(n_traces)]
    elif len(labels) != n_traces:
        raise ValueError(
            f"Number of labels ({len(labels)}) doesn't match number of traces ({n_traces})"
        )
    
    # Convert all traces
    traces_np = [_to_numpy(t) for t in traces]
    seq_len = traces_np[0].shape[0]
    num_classes = traces_np[0].shape[1]
    
    # Check which traces are deterministic vs stochastic
    traces_is_det = [_is_deterministic(t) for t in traces_np]
    
    # Generate activity names
    if activity_names is None:
        activity_names = [f"Class {i}" for i in range(num_classes)]
    
    # Generate colors
    if colors is None:
        # Use plotly's qualitative color palette
        if num_classes <= 10:
            color_sequence = px.colors.qualitative.Plotly
        elif num_classes <= 24:
            color_sequence = px.colors.qualitative.Dark24
        else:
            # Generate colors from HSL
            color_sequence = [
                f"hsl({int(360 * i / num_classes)}, 70%, 50%)"
                for i in range(num_classes)
            ]
        colors = [color_sequence[i % len(color_sequence)] for i in range(num_classes)]
    
    # Calculate height
    if height is None:
        height = n_traces * row_height + 100  # Extra space for title and legend
    
    fig = go.Figure()
    
    # Track which classes have been added to legend
    legend_added = set()
    
    # Process each trace
    for trace_idx, (trace_np, label, is_det) in enumerate(zip(traces_np, labels, traces_is_det)):
        class_indices, confidences = _trace_to_class_indices(trace_np)
        
        # Find segments (runs of same class) with confidence info
        segments = []
        start = 0
        current_class = class_indices[0]
        
        for t in range(1, seq_len):
            if class_indices[t] != current_class:
                # Calculate average confidence for segment
                seg_conf = np.mean(confidences[start:t])
                segments.append((start, t, current_class, seg_conf))
                start = t
                current_class = class_indices[t]
        # Add final segment
        seg_conf = np.mean(confidences[start:seq_len])
        segments.append((start, seq_len, current_class, seg_conf))
        
        # y position (from top to bottom)
        y_pos = n_traces - 1 - trace_idx
        
        # Add each segment as a bar
        for seg_start, seg_end, class_idx, avg_conf in segments:
            # Build hover text
            hover_parts = [
                f"<b>{label}</b>",
                f"Activity: {activity_names[class_idx]}",
                f"Time: {seg_start} - {seg_end}",
                f"Duration: {seg_end - seg_start}",
            ]
            # Add confidence info for stochastic traces
            if not is_det:
                hover_parts.append(f"Avg Confidence: {avg_conf:.3f}")
            
            # Only show legend once per class
            show_in_legend = class_idx not in legend_added
            if show_in_legend:
                legend_added.add(class_idx)
            
            fig.add_trace(go.Bar(
                x=[seg_end - seg_start],
                y=[label],
                base=seg_start,
                orientation="h",
                marker=dict(color=colors[class_idx]),
                name=activity_names[class_idx],
                legendgroup=str(class_idx),
                showlegend=show_in_legend,
                hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
        barmode="stack",
        xaxis=dict(
            title="Time Step",
            range=[0, seq_len],
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=labels[::-1],  # Reverse to match visual order
        ),
        width=width,
        height=height,
        legend=dict(
            title="Activities",
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
        ),
        bargap=0.1,
    )
    
    return fig
