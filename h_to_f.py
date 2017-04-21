import struct
#import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=str, help='benchmark')
#parser.add_argument('-t', type=str, help='type of the file: data or nn_config')
parser.add_argument('-a', type=str, help='which architecture: pipelined_vector, vector or systolic')
parser.add_argument('-f', type=str, help='filename: i_h1_h2_o')
args = parser.parse_args()

file_name = "benchmark/"+args.b+"/data/"+args.a+"/"+args.f

in_file = open(file_name+".dat", 'r')

# TODO: not finished. start from here
numbers = [[float(i) for i in j.split()] for j in in_file.readlines()]
#print numbers

in_file.close()

out_file = open(file_name+"_hex.dat", 'w')

for i in xrange(len(numbers)):
    for j in xrange(len(numbers[0])):
        #print len(numbers[0])
        #print numbers[i][j]
        #print struct.pack(">f", numpy.float32(numbers[i][j])).encode("hex")
        out_file.write(str(struct.pack(">f", numbers[i][j]).encode("hex")) + "\n")
    #out_file.write("\n")

out_file.close()
