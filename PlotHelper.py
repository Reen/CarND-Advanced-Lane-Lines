import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def render_row(fig, row, cols, titles):
    col = 1
    num_axes = len(fig.axes)
    num_rows = 0
    for i in range(num_axes):
        ax = fig.axes[i]
        num_rows = ax.numRows
        ax.change_geometry(ax.numRows + 1, cols, i + 1)
    
    for img in row:
        ax = fig.add_subplot(num_rows + 1, cols, num_axes + col)
        ax.imshow(img)
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[col - 1])
        col+=1
        


def plot_image_grid(images, titles, figsize=(20,10)):
    fig = plt.figure(figsize=figsize)
    first = True
    for row in images:
        cols = len(row)
        render_row(fig, row, cols, titles)
        if first:
            titles = None
            first = False
    
    return fig


