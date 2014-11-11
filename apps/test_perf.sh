#!/bin/bash

MB_SIZE=32
while [ $MB_SIZE -le 1024 ]; do
  /usr/bin/time -o log_perf -a -f "%e\t%C" ./cnn_multi -mb $MB_SIZE
  /usr/bin/time -o log_perf -a -f "%e\t%C" ./cnn -mb $MB_SIZE
  let MB_SIZE=2*MB_SIZE
done
