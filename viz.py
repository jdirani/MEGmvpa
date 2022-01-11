import matplotlib.pyplot as plt
import seaborn as sns


def plot_temporal_scores(scores, times, fname, chance_level=None, figsize=[12,4], title='Temporal decoding'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, scores, label='scores')
    if chance_level:
        ax.axhline(chance_level, color='k', linestyle='--', label='chance')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Score')
    ax.legend()
    ax.set_title(title)
    if fname:
        fig.savefig(fname)
        plt.close()



def plot_generalization_scores(scores, fname, times, vmin=0, vmax=1, annot=True, title='', num_ticks=20):
    '''
    annot : write values in matrix cells.
    '''
    fig, ax = plt.subplots(1)
    sns.heatmap(scores, annot=annot, cmap='YlOrRd', vmin=vmin, vmax=vmax,
                xticklabels=times, yticklabels=times,
                ax=ax)

    plt.locator_params(nbins=num_ticks)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") # to rotate ticks
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor") # to rotate ticks
    # ---- add titles
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(title)
    if fname:
        fig.savefig(fname)
        plt.close()
