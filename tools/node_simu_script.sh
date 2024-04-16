#!/bin/bash
set -x

func_num=810

## fixed alpha 
for ratio in 95 98; do
    for alpha in 0 1 2 4 8 10 20 40 80 100; do
        python3 node_simu.py -f $func_num -m 60 -t $ratio -a $alpha
        mv slo_${func_num}_60.npy slo_${ratio}_${alpha}.npy
        mv stat_${func_num}_60.npy stat_${ratio}_${alpha}.npy
        mv step_${func_num}_60.npy step_${ratio}_${alpha}.npy
    done
done

## fixed scalar and diff
# for ratio in 95; do
#     for scalar in 120 150 200; do
#         for diff in 1 2 4 6 8 10; do
#             python3 node_simu.py -f $func_num -m 60 -t $ratio -s $scalar -i $diff
#             mv slo_${func_num}_60.npy slo_${scalar}_${diff}.npy
#             mv stat_${func_num}_60.npy stat_${scalar}_${diff}.npy
#             mv step_${func_num}_60.npy step_${scalar}_${diff}.npy
#         done
#     done
# done

## fixed all
# for ratio in 95; do
#     for scalar in 120 140 160 180 200; do
#         for window in 10 20 40 60 80 100; do
#             for diff in 1 2 4 6 8 10; do
#                 python3 node_simu.py -f $func_num -m 60 -t $ratio -s $scalar  -i $diff -w $window
#                 # mv slo_${func_num}_60.npy slo_${scalar}_${window}_${diff}.npy
#                 # mv stat_${func_num}_60.npy stat_${scalar}_${window}_${diff}.npy
#                 mv step_${func_num}_60.npy step_${scalar}_${window}_${diff}.npy
#             done

#         done
#     done
# done