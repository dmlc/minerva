import sys,os
import struct
import array

in_file_name = sys.argv[1]
out_file_name = in_file_name + "1"
fin = open(in_file_name, 'rb')
fout = open(out_file_name, 'wb')

num_lines = struct.unpack('i', fin.read(4))[0]
line_len = struct.unpack('i', fin.read(4))[0]

fout.write(struct.pack('i', line_len))
fout.write(struct.pack('i', num_lines))

in_arr = array.array('f')
in_arr.fromfile(fin, num_lines * line_len)
for i in range(line_len):
    for j in range(num_lines):
        in_idx = j * line_len + i
        fout.write(struct.pack('f', in_arr[in_idx]))
