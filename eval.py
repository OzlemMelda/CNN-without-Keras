import pickle
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import glob


def scatter(x, labels):
    # Choose a color palette with seaborn
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = plt.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # Add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


if __name__ == '__main__':

    # Load Test Data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    # Load Final Model, Selected Test Data and Calculate Test Accuracy
    for file in glob.glob("*.pk"):
        pkl_filename = file

    with open(pkl_filename, 'rb') as file:
        final_model = pickle.load(file)
    test_loss, test_acc = final_model.check_accuracy(X_test, y_test)
    print('Test Accuracy: %.4f' % test_acc)

    """# Load trained model for TSNE
    RS = 123
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
                    
    for i in [1, 19, 20]:
        pkl_filename = 'results/final_model_ada_grad/model_' + str(i) + '.pk'

        with open(pkl_filename, 'rb') as file:
            final_model = pickle.load(file)
        final_model.fp(X_test, y_test, "check")
        tsne_1 = TSNE(random_state=RS).fit_transform(final_model.h1_history_x[0].reshape(10000, 8192))
        scatter(tsne_1, final_model.h_history_y[0])
        plt.show()
        tsne_2 = TSNE(random_state=RS).fit_transform(final_model.h2_history_x[0].reshape(10000, 4096))
        scatter(tsne_2, final_model.h_history_y[0])
        plt.show()
        tsne_3 = TSNE(random_state=RS).fit_transform(final_model.h3_history_x[0].reshape(10000, 2048))
        scatter(tsne_3, final_model.h_history_y[0])
        plt.show()
        tsne_4 = TSNE(random_state=RS).fit_transform(final_model.h4_history_x[0].reshape(10000, 1024))
        scatter(tsne_4, final_model.h_history_y[0])
        plt.show()
        tsne_5 = TSNE(random_state=RS).fit_transform(final_model.h5_history_x[0].reshape(10000, 512))
        scatter(tsne_5, final_model.h_history_y[0])
        plt.show()"""

