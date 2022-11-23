import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button
from matplotlib.path import Path


def zoom_factory(ax, base_scale=1.1):
    """
    parameters
    ----------
    ax : matplotlib axes object
        axis on which to implement scroll to zoom
    base_scale : float
        how much zoom on each tick of scroll wheel

    returns
    -------
    disconnect_zoom : function
        call this to disconnect the scroll listener
    """

    def limits_to_range(lim):
        return lim[1] - lim[0]

    fig = ax.get_figure()  # get the figure of interest
    toolbar = fig.canvas.toolbar
    toolbar.push_current()
    orig_xlim = ax.get_xlim()
    orig_ylim = ax.get_ylim()
    orig_yrange = limits_to_range(orig_ylim)
    orig_xrange = limits_to_range(orig_xlim)
    orig_center = ((orig_xlim[0] + orig_xlim[1]) / 2, (orig_ylim[0] + orig_ylim[1]) / 2)

    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        # set the range
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = base_scale
        elif event.button == 'down':
            # deal with zoom out
            #             if orig_xlim[0]<cur_xlim[0]
            scale_factor = 1 / base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
        #             print(event.button)
        # set new limits
        new_xlim = [xdata - (xdata - cur_xlim[0]) / scale_factor,
                    xdata + (cur_xlim[1] - xdata) / scale_factor]
        new_ylim = [ydata - (ydata - cur_ylim[0]) / scale_factor,
                    ydata + (cur_ylim[1] - ydata) / scale_factor]
        new_yrange = limits_to_range(new_ylim)
        new_xrange = limits_to_range(new_xlim)

        if np.abs(new_yrange) > np.abs(orig_yrange):
            new_ylim = orig_center[1] - new_yrange / 2, orig_center[1] + new_yrange / 2
        if np.abs(new_xrange) > np.abs(orig_xrange):
            new_xlim = orig_center[0] - new_xrange / 2, orig_center[0] + new_xrange / 2
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

        toolbar.push_current()
        ax.figure.canvas.draw_idle()  # force re-draw

    # attach the call back
    cid = fig.canvas.mpl_connect('scroll_event', zoom_fun)

    def disconnect_zoom():
        fig.canvas.mpl_disconnect(cid)

        # return the disconnect function

    return disconnect_zoom


class image_lasso_selector:
    def __init__(self, img, image_name, mask_path, mask_alpha=.75, figsize=(2, 2), zoom_scale=1.1):
        """
        img must have shape (X, Y, 3)
        """
        self.img = img
        self.image_name = '.'.join(image_name.split('.')[:-1]) + '_mask.png'
        self.mask_alpha = mask_alpha
        plt.ioff()  # see https://github.com/matplotlib/matplotlib/issues/17013
        self.fig = plt.figure(figsize=figsize, dpi=300)
        self.ax = self.fig.gca()
        self.displayed = self.ax.imshow(img)
        plt.ion()

        self.zoom_scale = zoom_scale

        lineprops = {'color': 'black', 'linewidth': 1, 'alpha': 0.8}
        self.lasso = LassoSelector(self.ax, self.onselect, lineprops=lineprops, useblit=False)
        self.lasso.set_visible(True)

        pix_x = np.arange(self.img.shape[0])
        pix_y = np.arange(self.img.shape[1])
        xv, yv = np.meshgrid(pix_y, pix_x)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T

        self.mask = np.zeros(self.img.shape[:2])
        self.mask_path = os.path.join(mask_path, self.image_name)
        self.mask_path = mask_path

        ax_button = plt.axes([0.1, 0.1, 0.2, 0.1])
        self.save_button = Button(ax_button, 'Save', color='white', hovercolor='grey')

        ax2_button = plt.axes([0.1, 0.5, 0.2, 0.1])
        self.reset_button = Button(ax2_button, 'Reset', color='white', hovercolor='grey')



    def save_mask(self, save_if_no_nonzero=False):
        """
        save_if_no_nonzero : boolean
            Whether to save if class_mask only contains 0s
        """
        i = 0
        os.makedirs(self.mask_path, exist_ok=True)
        iterator = os.path.join(self.mask_path, f'{str(i)}_{self.image_name}')
        while os.path.exists(iterator) is True:
            i += 1
            iterator = os.path.join(self.mask_path, f'{str(i)}_{self.image_name}')

        cv2.imwrite(filename=iterator, img=self.mask * 255)

    def reset_mask(self, save_if_no_nonzero=False):
        self.mask = np.zeros(self.img.shape[:2])
        self.displayed.set_data(self.img)
        self.fig.canvas.draw()

    def onselect(self, verts):
        self.verts = verts
        p = Path(verts)
        self.indices = p.contains_points(self.pix, radius=0).reshape(self.mask.shape)
        self.draw_with_mask()

    def draw_with_mask(self):
        array = self.displayed.get_array().data

        # https://en.wikipedia.org/wiki/Alpha_compositing#Straight_versus_premultiplied
        self.mask[self.indices] = 1
        c_overlay = self.mask[self.indices][..., None] * [1., 0, 0] * self.mask_alpha
        array[self.indices] = (c_overlay + self.img[self.indices] * (1 - self.mask_alpha))

        self.displayed.set_data(array)
        self.fig.canvas.draw_idle()

    def _ipython_display_(self):
        display(self.fig.canvas)