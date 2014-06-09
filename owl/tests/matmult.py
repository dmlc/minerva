#!/usr/bin/env python
# Copyright 2014 Project Athena

from libowl import Index
from libowl import constant_chunk

ia = Index(100, 200)
a = constant_chunk(ia, 0.2)
ib = Index(200, 50)
b = constant_chunk(ib, 0.1)
c = a * b
c.print_()
