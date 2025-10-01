import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import numpy as np

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]


def plot_figure(X, Y, label_x, label_y, title, path: str):

    os.makedirs(path, exist_ok=True)

    # Create a plot
    plt.plot(X, Y)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)

    # Save the plot as a PDF file
    plt.savefig(path + "plot.pdf")

    # Show the plot (optional)
    plt.show()


# plot_figure(x,y, 'X-axis', 'Y-axis', 'Sample Plot')


def plot_loss_accuracy_in_pdf(X, Y, loss, label_x, label_y, title, path: str):

    os.makedirs(path, exist_ok=True)

    # Create a plot
    plt.plot(X, Y, label="Accuracy")
    plt.plot(X, loss, label="Loss")
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    # Add a legend
    plt.legend()

    # Save the plot as a PDF file
    plt.savefig(path + "plot.pdf")

    # Show the plot (optional)
    plt.show()


def plot_data_dict_in_pdf(data: dict, path: str, title="Title", show=False):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(path, exist_ok=True)

    # Create a plot
    X = data["x"]
    file_name = data["short_name"]
    if data["dual_axis"]:
        fig, ax1 = plt.subplots()
        Y_1, Y_2 = data["y"]
        Legend_1, Legend_2 = data["legends"]
        label_x, (label_y1, label_y2) = data["labels"]

        ax1.set_xlabel(label_x)
        ax1.set_ylabel(label_y1, color="b")
        colors = ["b", "g", "r", "c", "m", "y", "k"]  # List of colors
        for i, (Y, legend) in enumerate(zip(Y_1, Legend_1)):
            ax1.plot(
                X, Y, label=legend, color=colors[i % len(colors)]
            )  # Use modulo to loop through colors
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.set_xlabel(label_x)
        ax2.set_ylabel(label_y2, color="r")
        for i, (Y, legend) in enumerate(zip(Y_2, Legend_2)):
            ax2.plot(
                X, Y, label=legend, color=colors[(i + len(Y_1)) % len(colors)]
            )  # Start from a new color
        ax2.legend(loc="lower right")
    max_acc = data["max_acc_g"]
    plt.title(title + f":max({max_acc})")

    # Save the plot as a PDF file
    plt.savefig(f"{path}{file_name}.pdf")

    # Show the plot (optional)
    if show:
        plt.show()


# Example usage:
# plot_data_dict_in_pdf(loaded_data)

def visualize_embeddings(features, labels, plot_path="./plot.pdf"):
    features = features[~np.isnan(features).any(axis=1)]
    num_samples = features.shape[0]
    # print(f"Samples no. : {num_samples}")
    if(num_samples<=1):
        return
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    perplexity = min(30, num_samples -1)  # Ensure perplexity is less than number of samples
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_features = tsne.fit_transform(features)
    # Create a custom colormap
    c_code ="tab20" if len(np.unique(labels))<=20 else "tab20c"
    # cmap = plt.get_cmap(c_code, len(np.unique(labels)))  # Get a colormap with as many unique colors as labels

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap=c_code)
    plt.legend(
        # handles=scatter.legend_elements()[0],
        # labels=[str(label) for label in np.unique(labels)],
        *scatter.legend_elements(),
        title="Classes",
        bbox_to_anchor=(0.005, 0.995), loc='upper left'
    )
    plt.title("t-SNE Visualization")
    plt.savefig(plot_path)
    plt.close()


