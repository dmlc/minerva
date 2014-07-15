# Copyright 2014 Project Athena

import libowl as lo


def constant_chunk(dimension, value):
    index = lo.Index(dimension[0], dimension[1])
    chunk = lo.constant_chunk(index, value)
    return chunk
