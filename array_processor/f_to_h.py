import struct
#import numpy

in_file = open("input.dat", 'r')

numbers = [[float(i) for i in j.split()] for j in in_file.readlines()]

in_file.close()

out_file = open("input_hex.dat", 'w')

for i in xrange(len(numbers)):
    for j in xrange(len(numbers[0])):
        #print numbers[i][j]
        #print struct.pack(">f", numpy.float32(numbers[i][j])).encode("hex")
        out_file.write(str(struct.pack(">f", numbers[i][j]).encode("hex")) + " ")
    out_file.write("\n")

out_file.close()
