#!/bin/bash

find minerva -type f -print0 | xargs -0 -I {} chmod 664 {}
find minerva -type d -print0 | xargs -0 -I {} chmod 775 {}
