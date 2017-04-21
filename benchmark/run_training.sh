#!/bin/bash

H1=0
H2=0
LEARNING_RATE=0.05
MAX_STEPS=4000
BENCHMARK=hotspot

for i in "$@"
do
    case $i in
        -h1=*|-hidden1=*)
            H1="${i#*=}"
            shift
        ;;
        -h2=*|-hidden2=*)
            H2="${i#*=}"
            shift
        ;;
        -b=*|-benchmark=*)
            BENCHMARK="${i#*=}"
            shift
        ;;
        -l=*|-learning_rate=*)
            LEARNING_RATE="${i#*=}"
            shift
        ;;
        -m=*|-max_steps=*)
            MAX_STEPS="${i#*=}"
            shift
        ;;
    esac
done

mkdir -p ${BENCHMARK}/data/train/train_result
mkdir -p ${BENCHMARK}/nn_config

python train_dnn.py --benchmark=$BENCHMARK \
    --learning_rate=$LEARNING_RATE --max_steps=$MAX_STEPS \
    --hidden1=$H1 --hidden2=$H2 \
    --data_dir=/home/cosine/spring2017/cs533/project/benchmark/${BENCHMARK}/data/train/ \
    --config_dir=/home/cosine/spring2017/cs533/project/benchmark/${BENCHMARK}/nn_config/

#case $BENCHMARK in 
#    hotspot)
#        python train_dnn.py --learning_rate=0.02 --max_steps=9100 \
#            --data_dir=/home/cosine/spring2017/cs533/project/benchmark/hotspot/data/train/ \
#            --config_dir=/home/cosine/spring2017/cs533/project/benchmark/hotspot/nn_config/
#    ;;
#    fft)
#        python train_dnn.py --learning_rate=0.05 --max_steps=4000 \
#            --data_dir=/home/cosine/spring2017/cs533/project/benchmark/fft/data/train/ \
#            --config_dir=/home/cosine/spring2017/cs533/project/benchmark/fft/nn_config/
#    ;;
#    inversek2j)
#        python train_dnn.py --learning_rate=0.05 --max_steps=4000 \
#            --data_dir=/home/cosine/spring2017/cs533/project/benchmark/inversek2j/data/train \
#            --config_dir=/home/cosine/spring2017/cs533/project/benchmark/inversek2j/nn_config/
#    ;;
#esac

