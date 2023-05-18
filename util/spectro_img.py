import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    fig.canvas.draw()
    return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))

def compute_visuals(sp=None, pha=None, abs=False):
    sp = sp.transpose() if sp is not None else None
    pha = pha.transpose() if pha is not None else None
    sp_spectro = None
    sp_hist = None
    if sp is not None:
        sp_fig, sp_ax = plt.subplots()
        sp_ax.pcolormesh(sp if not abs else np.abs(sp), cmap='PuBu_r')
        sp_spectro = fig2img(sp_fig)

        sp_hist_fig, sp_hist_ax = plt.subplots()
        sp_hist_ax.hist(sp.reshape(-1,1),bins=100)
        sp_hist = fig2img(sp_hist_fig)

    if pha is not None:
        pha_fig, pha_ax = plt.subplots()
        pha_ax.pcolormesh(pha, cmap='cool')
        pha = fig2img(pha_fig)

    plt.close('all')
    return sp_spectro, sp_hist, pha
