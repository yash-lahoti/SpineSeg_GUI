import argparse
import json
import os
import sys
import matplotlib.pyplot as plt
import pylab
import cv2
from utils import *

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required args
    parser.add_argument('--data-dir',
                        required=True,
                        help='config.json')
    parser.add_argument('--save-dir',
                        required=True,
                        help='config.json')
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    cwd = os.getcwd()

    image_dir = args.data_dir
    image_paths = os.listdir(image_dir)

    for img in image_paths:
        if '.png' in img.lower() or '.jpg' in img.lower():
            img_loaded = cv2.imread(os.path.join(image_dir, img))
            obj = image_lasso_selector(img_loaded, image_name=img, mask_path=args.save_dir)
            obj.save_button.on_clicked(obj.save_mask)
            obj.reset_button.on_clicked(obj.reset_mask)
            # gotta do this after creating the image, and the image needs to come after all the buttons
            if obj.zoom_scale is not None:
                obj.disconnect_scroll = zoom_factory(obj.ax, base_scale=obj.zoom_scale)

            #plt.rcParams["figure.figsize"] = (20,3)
            #plt.rcParams["figure.autolayout"] = False
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.autoscale()



            pylab.ion()
            obj.fig.show()
            plt.show(block=True)
            del obj

if __name__ == '__main__':
    main(sys.argv[1:])
