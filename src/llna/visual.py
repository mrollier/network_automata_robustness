import numpy as np
import matplotlib.pyplot as plt
import igraph as ig

import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from IPython.display import HTML
from matplotlib import rcParams
rcParams['animation.embed_limit'] = 256  # Set a higher limit for embedding animations

# %% COLOURS

# https://styleguide.ugent.be/basic-principles/colours.html
def get_ugent_colors_dict():
    ugent_colors = {
    'ugent_blue': '#1E64C8',
    'ugent_yellow': '#FFD200',
    'ugent_white': '#FFFFFF',
    'ugent_black': '#000000',
    'lw_yellow': '#F1A42B',     # Faculty of Arts and Philosophy
    're_red': '#DC4E28',        # Faculty of Law and Criminology
    'we_aqua': '#2D8CA8',       # Faculty of Sciences
    'ge_pink': '#E85E71',       # Faculty of Medicine and Health Sciences
    'ea_blue': '#8BBEE8',       # Faculty of Engineering and Architecture
    'eb_green': '#AEB050',      # Faculty of Economics and Business Administration
    'di_purple': '#825491',     # Faculty of Veterinary Medicine
    'pp_orange': '#FB7E3A',     # Faculty of Psychology and Educational Sciences
    'bw_turquoise': '#27ABAD',  # Faculty of Bioscience Engineering
    'fw_purple': '#BE5190',     # Faculty of Pharmaceutical Sciences
    'ps_green': '#71A860'       # Faculty of Political and Social Sciences
    }
    return ugent_colors

def get_ugent_cmap():
    ugent_colors_dict = get_ugent_colors_dict()
    ugent_white = ugent_colors_dict['ugent_white']
    ugent_blue = ugent_colors_dict['ugent_blue']
    cmap = ListedColormap([ugent_white, ugent_blue])  # 0: white, 1: blue
    return cmap

# %% ANIMATIONS

def make_animation_object(grids, title=None, cmap='Greys'):
    """
    Creates an animation object for visualizing a sequence of 2D grids.

    Parameters:
    -----------
    grids : numpy.ndarray
        A 3D array where the first dimension represents time steps, and the 
        subsequent dimensions represent the 2D grid at each time step.
    title : str, optional
        The title to display on the animation frames. Default is None.
    cmap : str, optional
        The colormap to use for displaying the grids. Default is 'Greys'.

    Returns:
    --------
    matplotlib.animation.FuncAnimation
        An animation object that can be saved or displayed using matplotlib's
        animation tools.
    """
    # find number of time steps
    T = grids.shape[0]
    # Set up figure
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grids[0], cmap=cmap)
    # change spine width
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # Update function for animation
    def update(frame):
        im.set_array(grids[frame])
        ax.set_title(title, fontsize=20)
        ax.set_xticks([]); ax.set_yticks([])
        return [im]
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=T, interval=100)
    plt.close()
    return ani

def make_animation_density_evolution(timesteps, configs, title=None, plotcolor='k', vertical_color='k'):
    """
    Creates an animated plot showing the evolution of the global state density over time.

    Parameters:
    -----------
    timesteps : array-like
        A sequence of time steps corresponding to the simulation.
    configs : numpy.ndarray
        A 3D array where each element represents the configuration of the system at a given time step.
        The shape of the array should be (T, H, W), where T is the number of time steps, and H and W
        are the height and width of the configuration grid.
    title : str, optional
        The title of the plot. If None, no title is displayed. Default is None.
    plotcolor : str, optional
        The color of the line plot showing the mean global state density. Default is 'k' (black).
    vertical_color : str, optional
        The color of the vertical line and scatter point indicating the current time step. Default is 'k' (black).
    Returns:
    --------
    matplotlib.animation.FuncAnimation
        An animation object that can be displayed or saved as a video.
    Notes:
    ------
    - The global state density is calculated as the mean of the `configs` array along the last two axes.
    - The animation updates a scatter point and a vertical line to indicate the current time step.
    - The x-axis and y-axis ticks are removed for a cleaner visualization.
    - The y-axis is labeled as "Global state density" with a fixed range of [-0.05, 1.05].
    """
    # find number of time steps
    T = len(timesteps)
    # calculate means
    config_means = np.mean(configs, axis=(1,2))
    # Set up figure
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(timesteps, config_means, color=plotcolor, linewidth=2, zorder=0)
    sc = ax.scatter(timesteps[0], config_means[0], color=vertical_color, s=100, zorder=1)
    vline = ax.axvline(x=timesteps[0], color=vertical_color, linestyle='--', linewidth=2, zorder=2)
    # change spine width
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # Update function for animation

    # Add title
    if title is not None:
        ax.set_title(title, fontsize=20)

    # Update function
    def update(frame):
        # Update scatter point position
        sc.set_offsets([[timesteps[frame], config_means[frame]]])
        # Move vertical line
        vline.set_xdata([timesteps[frame]])
        # ticks and range
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-0.05,1.05])
        ax.set_ylabel(f"Global state density", fontsize=35)
        return [sc]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=T, interval=100)
    plt.close()
    return ani

def make_igraph_animation(graph, vertex_colors_over_time, vertex_size=25, layout=None, title=None, interval=300):
    """
    Creates an animation of an igraph graph with changing vertex colors.

    Parameters:
    -----------
    graph : igraph.Graph
        The igraph graph object to animate.
    vertex_colors_over_time : list or numpy.ndarray
        A 2D list or array where each row represents the vertex colors at a specific frame.
    vertex_size : int, optional
        The size of the vertices in the graph. Default is 25.
    layout : igraph.Layout, optional
        The layout to use for positioning the vertices. If None, the Fruchterman-Reingold layout is used.
    title : str, optional
        The title to display on the animation frames. Default is None.
    interval : int, optional
        The time interval between frames in milliseconds. Default is 300.

    Returns:
    --------
    matplotlib.animation.FuncAnimation
        An animation object that can be displayed or saved as a video.
    """
    if layout is None:
        layout = graph.layout("fr")  # Fruchterman-Reingold layout

    num_frames = len(vertex_colors_over_time)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_obj = None

    def update(frame):
        nonlocal plot_obj
        ax.clear()

        # Set current vertex colors
        graph.vs["color"] = vertex_colors_over_time[frame]

        # Draw the graph
        plot_obj = ig.plot(
            graph,
            target=ax,
            layout=layout,
            vertex_size=vertex_size,
            vertex_label=None,
            vertex_frame_width=2,
            edge_width=2
        )
        if title:
            ax.set_title(f"{title} (Frame {frame})", fontsize=16)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        return plot_obj,

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval)
    plt.close()
    return ani

def add_pauses(grids, pause_every=8, pause_length=5):
    """
    Repeat frames at every multiple of `pause_every` to simulate a pause.

    Args:
        grids (list): List of 2D arrays representing frames.
        pause_every (int): Pause at frames divisible by this number.
        pause_length (int): How many extra times to repeat the paused frame.

    Returns:
        List of grids with pauses added.
    """
    new_grids = []
    for i, frame in enumerate(grids):
        new_grids.append(frame)
        if i % pause_every == 0:
            for _ in range(pause_length):
                new_grids.append(frame)
    return np.array(new_grids)

def get_heart_shaped_grid():
    """
    Generate a 2D numpy array representing a heart-shaped grid.

    Returns:
        numpy.ndarray: A binary grid (1s and 0s) in the shape of a heart.
    """
    heart_shape = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    return heart_shape
