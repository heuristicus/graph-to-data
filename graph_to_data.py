#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import argparse

WHITE_PIXEL = np.array([1, 1, 1])
BLACK_PIXEL = np.array([0, 0, 0])

class GraphToData(object):

    def __init__(self, args):
        self.args = args
        img = mpimg.imread(args.infile)
        # These are the minimum and maximum values on the axes of the graph
        self.xminmax = (args.xrange[0], args.xrange[1])
        self.yminmax = (args.yrange[0], args.yrange[1])

        # Before the transpose, we would read from left to right on the graph. When
        # you transpose, the effect is that each row is now a column. This means
        # what was previously the zeroth column is now the zeroth row, i.e. the left
        # part of the graph is now the top. What was the zeroth row is now the
        # zeroth column, i.e. the top row has moved to the left column. This means
        # that reading each row is actually looking down each column of the original
        # graph.
        self.img = np.transpose(img, [1,0,2])
        self.imsize = (len(self.img[0]), len(self.img))

    def process(self):
        line_extents = self.capture_line_extent()
        graph_data = self.line_extents_to_data(line_extents)
        if self.args.outfile:
            self.to_file(graph_data, self.args.outfile)

    def capture_line_extent(self):
        """Scan through each row of the image and find the first and last black pixels
        in a series of consecutive pixels. We assume that there is only one such
        area in the row, and that all black pixels in the row exist in that area.

        """
        pixels = self.imsize[0] * self.imsize[1]
        # Print information once we process this percentage of the pixels in the image
        processed = 0
        previous_percent = 0

        # Store the first and last appearances of black pixels in each row, where
        # the first black pixel is the first black pixel preceded by consecutive
        # white pixels, and where the last black pixel is the one where consecutive
        # black pixels are followed by a white pixel.
        line_capture = []
        # The x and y here correspond to the correct x and y for the transposed
        # image, not the values for the original image.
        for y, row in enumerate(self.img):
            # Keep track of the transitions between white and black pixels. We 
            previous_col = 'empty'
            first_black = 0 # first column in which we saw a black
            last_black = 0 # last column in which we saw a black pixel
            for x, col in enumerate(row):
                if (col == WHITE_PIXEL).all(): # This is a white pixel
                    if previous_col == 'black': # white pixel after black
                        # img[y][x - 1] = np.array([1,0,0]) # colour the last one red
                        last_black = x - 1
                        # break here so we don't look further than needed
                        break
                    previous_col = 'white'
                elif (col == BLACK_PIXEL).all():
                    if previous_col == 'white' or previous_col == 'empty': # black pixel after white
                        # img[y][x] = np.array([0,1,0]) # colour the first one green
                        first_black = x
                    previous_col = 'black'
                else:
                    # shouldn't get here, because the image should be binary
                    print("The image should be binary, but the pixel at ({0}, {1}) was {2}".format(y, x, col))

            # if last_black is zero, then it hit the bottom of the plotting area
            if last_black == 0 and first_black != 0:
                last_black = len(row)

            line_capture.append((first_black, last_black))

            processed += len(row)
            processed_percent = int((processed/float(pixels))*100)
            if processed_percent % 10 == 0 and previous_percent != processed_percent:
                print("Processed {0}% of the image.".format(processed_percent))
            previous_percent = processed_percent

        return line_capture

    def line_extents_to_data(self, extents):
        """Convert a list of tuples containing the first and last black pixels in a row
        into a list of floats with the approximate data value used to generate
        the graph image in each row.

        """
        # Assume that the actual value is the average of the first and last black
        # pixel. The values here are the inverted y coordinate of the data used to
        # plot that point. (or x, since we transposed the image)
        stripped_line = map(lambda v: self.imsize[0] - (v[0] + v[1])/2, extents)

        # plt.plot(stripped_line)
        # plt.ylim(0, imsize[0])
        # plt.xlim(0, imsize[1])
        # plt.show()

        graph_values = []
        x_values = np.arange(self.xminmax[0], self.xminmax[1], (self.xminmax[1]-self.xminmax[0])/len(stripped_line))
        # Get the scale between the pixel values (0 to max y) and the values in the graph (min y to max y)
        y_range = self.yminmax[1] - self.yminmax[0]
        pixel_to_graph_scale = y_range/self.imsize[0]
        # transform the each y-coordinate into the graph scale using the scale
        # factor, then add the min y value of the axis in the graph to get it
        # into the correct range.

        for ind, ycoord in enumerate(stripped_line):
            yvalue = ycoord * pixel_to_graph_scale + self.yminmax[0]
            graph_values.append(yvalue)

        plt.plot(x_values, graph_values)
        plt.ylim(self.yminmax[0], self.yminmax[1])
        plt.show()

        # plt.imshow(img)
        # plt.show()
        return {'xvals': x_values, 'yvals': graph_values}

    def to_file(self, lines, outfile):
        separator = self.args.separator or ','
        with open(outfile, 'w') as f:
            f.write(separator.join(['x', 'y']) + "\n")
            for ind, yvalue in enumerate(lines['yvals']):
                f.write(separator.join([str(lines['xvals'][ind]), str(yvalue)]) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process an image of a graph to extract the "\
                                     "data used to generate it")
    parser.add_argument('--xrange', nargs=2, metavar=('xmin', 'xmax'), type=float,
                        required=True,
                        help='Range of the x axis on the graph.')
    parser.add_argument('--yrange', nargs=2, metavar=('ymin', 'ymax'), type=float,
                        required=True,
                        help='Range of the y axis on the graph.')
    parser.add_argument('-i', metavar='infile', dest='infile', required=True,
                        help="The input is assumed to be a binary RGB image in which black is the"\
                        " plotted data. The image should be cropped such that the size of the image"\
                        " corresponds to the bounds of the plot, so zero on the x is the leftmost"\
                        " plot value, and the maximum y value is the maximum plot value."\
                        " There should only be a single line in the image.")
    parser.add_argument('-o', metavar='outfile', dest='outfile')
    parser.add_argument('--separator', help="Separator to use in the output file. Default is ','")

    graph_to_data = GraphToData(parser.parse_args())
    graph_to_data.process()
