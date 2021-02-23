# !/bin/bash

rm ./out/candy.png
rm ./out/jerry.png
rm ./out/large-candy.png
rm ./execs/lab5

bash compile.sh

srun -n 1 --gres=gpu:1 ./execs/lab5 /home/pp20/share/lab4/testcases/candy.png ./out/candy.png
png-diff ./out/candy.png /home/pp20/share/lab4/testcases/candy.out.png

srun -n 1 --gres=gpu:1 ./execs/lab5 /home/pp20/share/lab4/testcases/jerry.png ./out/jerry.png
png-diff ./out/jerry.png /home/pp20/share/lab4/testcases/jerry.out.png

srun -n 1 --gres=gpu:1 ./execs/lab5 /home/pp20/share/lab4/testcases/large-candy.png ./out/large-candy.png
png-diff ./out/large-candy.png /home/pp20/share/lab4/testcases/large-candy.out.png