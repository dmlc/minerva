#!/usr/bin/env python
# Copyright 2014 Project Athena

import owl

a = owl.constant_chunk((100, 200), 0.2)
b = owl.constant_chunk((200, 50), 0.1)
c = a * b
a.print_()
c.print_()
c.print_()
