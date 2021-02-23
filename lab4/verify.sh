# !/bin/bash

rm ./out/candy.png
rm ./execs/lab4

bash compile.sh

srun -n 1 --gres=gpu:1 ./execs/lab4 /home/pp20/share/lab4/testcases/candy.png ./out/candy.png
png-diff ./out/candy.png /home/pp20/share/lab4/testcases/candy.out.png

srun -n 1 --gres=gpu:1 ./execs/lab4 /home/pp20/share/lab4/testcases/candy.png ./out/candy.png
png-diff ./out/candy.png /home/pp20/share/lab4/testcases/candy.out.png