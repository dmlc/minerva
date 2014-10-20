#!/bin/bash

find minerva -type f -print0 | xargs -0 -I {} chmod 644 {}
