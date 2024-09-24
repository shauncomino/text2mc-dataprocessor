from typing import List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import offsetbox
import numpy as np


class ImageAnnotations2D:
    def __init__(self, xy, imgs: List[np.ndarray], labels: List[str], ax2d: Axes, figure: Figure):
        """
        Initialize with 2D positions, images, labels, and 2D axes.
        :param xy: A list of 2D coordinates (x, y)
        :param imgs: A list of images (numpy arrays)
        :param labels: A list of labels (strings)
        :param ax2d: 2D matplotlib axes
        :param figure: Matplotlib figure
        """
        self.xy = xy
        self.imgs = imgs
        self.labels = labels
        self.ax2d = ax2d
        self.figure = figure
        self.annot = []
        
        # Create annotations for each image and label at the given positions
        for xy, im, label in zip(self.xy, self.imgs, self.labels):
            self.annot.append(self.image(im, xy))
            self.annot.append(self.label(label, xy))
        
        # Set up canvas events (if needed for future enhancements)
        self.cid = self.ax2d.figure.canvas.mpl_connect("draw_event", self.update)

    def image(self, arr, xy):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=0.25)
        ab = offsetbox.AnnotationBbox(im, xy, pad=0)
        self.ax2d.add_artist(ab)
        return ab

    def label(self, label, xy):
        """ Place a label at position xy """
        text = offsetbox.TextArea(label, textprops={'fontsize': 4})  # Adjust fontsize if needed
        ab = offsetbox.AnnotationBbox(text, xy, xybox=(0, 16),
                                      xycoords='data',
                                      boxcoords="offset points")
        self.ax2d.add_artist(ab)
        return ab

    def update(self, event):
        """ Update annotations on draw event """
        # In 2D, no need to update projections, just redraw as is.
        for ab in self.annot:
            ab.xy = ab.xy  # This can be extended for dynamic updates if needed
