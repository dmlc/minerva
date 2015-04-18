#!/bin/bash

PYTHON=

if [ `command -v ipython >/dev/null 2>&1` ]; then
  PYTHON=python
else
  PYTHON=ipython
fi

$PYTHON -i owl/minerva_start.py
