import struct
#import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=str, help='benchmark')
parser.add_argument('-a', type=str, help='which architecture: pipelined_vector, vector or systolic')
parser.add_argument('-t', type=str, help='topology: i_h1_h2_o')
args = parser.parse_args()

#### input f to h ####
file_name = "benchmark/"+args.b+"/data/"+args.a+"/input"

in_file = open(file_name+".dat", 'r')
for i in xrange(1):
    in_file.readline()

numbers = [[float(i) for i in j.split()] for j in in_file.readlines()]
#print numbers

in_file.close()

out_file = open(file_name+"_hex.dat", 'w')

for i in xrange(len(numbers)):
#for i in xrange(1):
    for j in xrange(len(numbers[0])):
        #print len(numbers[0])
        #print numbers[i][j]
        #print struct.pack(">f", numpy.float32(numbers[i][j])).encode("hex")
        out_file.write(str(struct.pack(">f", numbers[i][j]).encode("hex")) + "\n")
    #out_file.write("\n")

out_file.close()


#### config f to h ####
file_name = "benchmark/"+args.b+"/nn_config/"+args.a+"/"+args.t

in_file = open(file_name+".dat", 'r')

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
