#!/usr/bin/env python

## MIT License
## (C) 2013 James Crooks

'''Short script to parse loop data from the tcl script dump
into a more managable form.  From the commandline this
script dumps the parsed form to the data to be collected
or operated on by Unix tools.
This script is intended only for use in the TCR project.'''

import sys
import numpy as np

def parse_data(filename):
    data_file = open(filename, 'r')
    raw_data = data_file.readlines()
    data_file.close()
    raw_xs = raw_data[0:len(raw_data):3]
    raw_ys = raw_data[1:len(raw_data):3]
    raw_zs = raw_data[2:len(raw_data):3]
    raw_coords = zip(raw_xs, raw_ys, raw_zs)
    split_data = []
    for frame in raw_coords:
        x_vals = map(float, frame[0].split())
        y_vals = map(float, frame[1].split())
        z_vals = map(float, frame[2].split())
        split_data.append(x_vals + y_vals + z_vals)
    return np.array(split_data)

if __name__ == "__main__":
    try:
        # This is probably not going to work the way I want...
        print parse_data(sys.argv[1])
    
    except:
        print "File not found"
        sys.exit(0)
