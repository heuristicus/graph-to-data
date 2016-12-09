#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys

WHITE_PIXEL = np.array([1, 1, 1])
BLACK_PIXEL = np.array([0, 0, 0])

if __name__ == '__main__':
    # The input is assumed to be a binary RGBA image in which black is the
    # plotted data. The image should be cropped such that the size of the image
    # corresponds to the bounds of the plot, so zero on the x is the leftmost
    # plot value, and the maximum y value is the maximum plot value.
    img = mpimg.imread(sys.argv[1])
    # These are the minimum and maximum values on the axes of the graph
    xminmax = (float(sys.argv[2]), float(sys.argv[3]))
    yminmax = (float(sys.argv[4]), float(sys.argv[5]))

    # Before the transpose, we would read from left to right on the graph. When
    # you transpose, the effect is that each row is now a column. This means
    # what was previously the zeroth column is now the zeroth row, i.e. the left
    # part of the graph is now the top. What was the zeroth row is now the
    # zeroth column, i.e. the top row has moved to the left column. This means
    # that reading each row is actually looking down each column of the original
    # graph.
    img = np.transpose(img, [1,0,2])
    imsize = (len(img[0]), len(img))

    print(imsize)
    pixels = imsize[0] * imsize[1]
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
    for y, row in enumerate(img):
        # Keep track of the transitions between white and black pixels. We 
        previous_col = "empty"
        first_black = 0 # first column in which we saw a black
        last_black = 0 # last column in which we saw a black pixel
        for x, col in enumerate(row):
            if (col == WHITE_PIXEL).all(): # This is a white pixel
                if previous_col == "black": # white pixel after black
                    # img[y][x - 1] = np.array([1,0,0]) # colour the last one red
                    last_black = x - 1
                    # break here so we don't look further than needed
                    break
                previous_col = "white"
            elif (col == BLACK_PIXEL).all():
                if previous_col == "white" or previous_col == "empty": # black pixel after white
                    # img[y][x] = np.array([0,1,0]) # colour the first one green
                    first_black = x
                previous_col = "black"
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

    # Assume that the actual value is the average of the first and last black
    # pixel. The values here are the inverted y coordinate of the data used to
    # plot that point. (or x, since we transposed the image)
    stripped_line = map(lambda v: imsize[0] - (v[0] + v[1])/2, line_capture)

    # plt.plot(stripped_line)
    # plt.ylim(0, imsize[0])
    # plt.xlim(0, imsize[1])
    # plt.show()
    
    graph_values = []
    # Output the approximated data, computed by using the input values for the
    # axis limits.
    with open("testout.txt", "w") as f:
        # Get the scale between the pixel values (0 to max y) and the values in the graph (min y to max y)
        y_range = yminmax[1] - yminmax[0]
        pixel_to_graph_scale = y_range/imsize[0]
        # transform the each y-coordinate into the graph scale using the scale
        # factor, then add the min y value of the axis in the graph to get it
        # into the correct range.
        for ycoord in stripped_line:
            yvalue = ycoord * pixel_to_graph_scale + yminmax[0]
            graph_values.append(yvalue)

    x_values = np.arange(xminmax[0], xminmax[1], (xminmax[1]-xminmax[0])/len(graph_values))
    plt.plot(x_values, graph_values)
    plt.ylim(yminmax[0], yminmax[1])
    plt.show()

    # plt.imshow(img)
    # plt.show()
