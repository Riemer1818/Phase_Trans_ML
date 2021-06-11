#1: number of temperatures
#2: number of states saved per temperature last 50%
#3: states per temperature (how many times we run the similation total)
#4: grid length
#5: ferromagnetic (1) antiferromagnetic (-1)
#6: number of itirations per atom on average
#7: output directory

julia ~/Phase_Trans_ML/make_data_2D.jl 100 10 500 50 1 25 ./Phase_Trans_ML/train_data
