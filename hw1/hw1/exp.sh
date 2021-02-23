# !bin/bash

echo -e "Start Experiments"

echo -e "Compiling hw1"
make
echo -e "hw1 Compiled"

exp_testcase=35
rep_f_name="report"
rep_f_format=".csv"
exp_n=536869888
exp_nodes=(1 2 3 4)
exp_procs=(1 2 4 8 12 16 24 48)
exps=1

echo -e "Experiment1 on testcase${exp_testcase}"
echo -e "nodes, processes, cpu, cpu_sort, cpu_memory, cpu_merge, I/O, I/O_read, I/O_write, communication, total, execution_time" >> measure/exp.csv
for node in "${exp_nodes[@]}"
do
    for proc in "${exp_procs[@]}"
    do  
        if [ $node -gt $proc ]
        then
            continue
        fi

        echo -e "Experiment ${exps}: $node nodes $proc procs"
        exp_res=`srun -n$proc -N$node ./hw1 $exp_n /home/pp20/share/hw1/testcases/$exp_testcase.in ./out/$exp_testcase.out measure/$rep_f_name$proc-$node$rep_f_format`
        echo "${node}, ${proc}, ${exp_res}" >> measure/exp.csv
        exps=$((exps+1))
    done
    echo -e "" >> measure/exp.csv
done