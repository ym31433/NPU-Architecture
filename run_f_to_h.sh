#!/bin/bash
# this script calls f_to_h twice to convert both the weight file and the input file to hex files

if [ $# != 3 ]
then
    echo "usage: ./run_f_to_h.sh -b=<benchmark> -a=<architecture> -t=<topology>"
    exit 1
fi

for i in "$@"
do
    case $i in
        -b=*|-benchmark=*)
            BENCHMARK="${i#*=}"
            shift
        ;;
        -a=*|-architecture=*)
            ARCHITECTURE="${i#*=}"
            shift
        ;;
        -t=*|-topology=*)
            TOPOLOGY="${i#*=}"
            shift
        ;;
    esac
done

mkdir -p benchmark/$BENCHMARK/data/$ARCHITECTURE
mkdir -p benchmark/$BENCHMARK/nn_config/$ARCHITECTURE

cp benchmark/$BENCHMARK/data/train/input.txt benchmark/$BENCHMARK/data/$ARCHITECTURE/input.dat
cp benchmark/$BENCHMARK/nn_config/${TOPOLOGY}.txt benchmark/$BENCHMARK/nn_config/$ARCHITECTURE/${TOPOLOGY}.dat

python f_to_h.py -t=data -f=input -b=$BENCHMARK -a=$ARCHITECTURE
python f_to_h.py -t=nn_config -f=$TOPOLOGY -b=$BENCHMARK -a=$ARCHITECTURE
