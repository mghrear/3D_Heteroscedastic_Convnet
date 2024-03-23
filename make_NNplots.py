import matplotlib.pyplot as plt
import numpy as np

def plot_HSCDC(name = "HSCDC_plot.pdf", split_space = 5, line_width = 1, neuron_size = 200):

    labels = [  "2160 neurons, flattened input",
                "500 neurons, relU activation",
                "200 neurons, relU activation",
                "50 neurons, tanh activation",
                "3 neurons, L2 norm activation",
                "1 neuron, softplus activation"]

    colors = [  "darkgrey",
                "steelblue",
                "goldenrod",
                "lightcoral",
                "darkmagenta",
                "darkseagreen"]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define the layers
    layers = [11, 9, 7, 5, 3]  # Input layer, hidden layer, and two output heads

    # Define the positions of layers
    layer_positions = np.arange(len(layers))

    # Create the nodes for each layer
    for i, num_nodes in enumerate(layers):

        if i <= 1:
            y_positions = np.arange(num_nodes) - num_nodes/2.0
            x_positions = np.ones_like(y_positions) * i
            ax.scatter(x_positions, y_positions,zorder=2,  s=neuron_size, label=labels[i],color=colors[i])
        elif i< 4:
            y_positions = np.arange(num_nodes) - num_nodes/2.0 + split_space
            x_positions = np.ones_like(y_positions) * i
            ax.scatter(x_positions, y_positions,zorder=2, s=neuron_size,label=labels[i],color=colors[i])
            ax.scatter(x_positions, -1.0*y_positions,zorder=2, s=neuron_size,color=colors[i])
        else:
            y_positions = np.arange(num_nodes) - num_nodes/2.0 + split_space
            x_positions = np.ones_like(y_positions) * i
            ax.scatter(x_positions, y_positions,zorder=2, s=neuron_size, label=labels[i],color=colors[i])
            ax.scatter(x_positions[1], -1.0*y_positions[1],zorder=2, s=neuron_size, label=labels[i+1],color=colors[i+1])

    # Connect the nodes with lines
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i + 1]):
                if i<1:
                    ax.plot([i, i + 1], [j - layers[i]/2.0, k- layers[i+1]/2.0], 'k',zorder=1, lw=line_width)
                elif i ==1:
                    ax.plot([i, i + 1], [j - layers[i]/2.0  , k- layers[i+1]/2.0 + split_space], 'k',zorder=1, lw=line_width)
                    ax.plot([i, i + 1], [-1.0*j + layers[i]/2.0 -1 , -1.0*k + layers[i+1]/2.0 - split_space], 'k',zorder=1, lw=line_width)
                elif i<3: 
                    ax.plot([i, i + 1], [j - layers[i]/2.0 + split_space , k- layers[i+1]/2.0 + split_space], 'k',zorder=1, lw=line_width)
                    ax.plot([i, i + 1], [-j + layers[i]/2.0 - split_space , -k + layers[i+1]/2.0 - split_space], 'k',zorder=1, lw=line_width)
                else:
                    ax.plot([i, i + 1], [j - layers[i]/2.0 + split_space , k- layers[i+1]/2.0 + split_space], 'k',zorder=1, lw=line_width)
                    if k==1:
                        ax.plot([i, i + 1], [-j + layers[i]/2.0 - split_space , -k + layers[i+1]/2.0 - split_space], 'k',zorder=1, lw=line_width)

    # Add a legend
    #ax.legend(loc = "lower left", prop={'size': 16})

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout()

    # Save the figure
    plt.savefig(name)

def plot_reg(name = "reg_plot.pdf", line_width = 1, neuron_size = 200):

    labels = [  "2160 neurons, flattened input",
                "500 neurons, relU activation",
                "200 neurons, relU activation",
                "50 neurons, tanh activation",
                "3 neurons, L2 norm activation",
                "1 neuron, softplus activation"]

    colors = [  "darkgrey",
                "steelblue",
                "goldenrod",
                "lightcoral",
                "darkmagenta",
                "darkseagreen"]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define the layers
    layers = [11, 9, 7, 5, 3]  # Input layer, hidden layer, and two output heads

    # Define the positions of layers
    layer_positions = np.arange(len(layers))

    # Create the nodes for each layer
    for i, num_nodes in enumerate(layers):

        y_positions = np.arange(num_nodes) - num_nodes/2.0
        x_positions = np.ones_like(y_positions) * i
        ax.scatter(x_positions, y_positions,zorder=2,  s=neuron_size, label=labels[i],color=colors[i])

    # Connect the nodes with lines
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i + 1]):
                ax.plot([i, i + 1], [j - layers[i]/2.0, k- layers[i+1]/2.0], 'k',zorder=1, lw=line_width)

    #This is just to update the legend
    ax.scatter([], [],zorder=2, s=neuron_size, label=labels[5],color=colors[5])


    # Add a legend
    ax.legend(loc = 0, prop={'size': 20})

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout()


    # Save the figure
    plt.savefig(name)

plot_reg()
plot_HSCDC()
