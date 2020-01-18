"""
This is for creating job files for ISYE clusters.
"""

cmd_name = input('cmd file name: ')
function_name = input('function_name: ')
seed = input('seed: ')
dim_w = input('dim_w: ')
filename = input('output filename: ')
iterations = input('iterations: ')
args = ' ' + function_name + ' ' + seed + ' ' + dim_w + ' ' + filename + ' ' + iterations
args_ = '_' + function_name + '_' + seed + '_' + dim_w + '_' + filename + '_' + iterations

file = open(cmd_name + '.cmd', 'w')
lines = list()
lines.append('universe=vanilla')
lines.append('getenv=true')
lines.append('executable=$ENV(HOME)/env/bin/python')
lines.append('+Requirements=((Machine=="isye-wilkins1.isye.gatech.edu")||(Machine=="isye-wilkins2.isye.gatech.edu")||(Machine=="isye-ames.isye.gatech.edu")||(Machine=="isye-fisher1.isye.gatech.edu")||(Machine=="isye-fisher2.isye.gatech.edu")||(Machine=="isye-fisher3.isye.gatech.edu")||(Machine=="isye-fisher4.isye.gatech.edu")||(Machine=="isye-wilkins3.isye.gatech.edu")||(Machine=="isye-wilkins4.isye.gatech.edu")||(Machine=="isye-jacobi1.isye.gatech.edu")||(Machine=="isye-jacobi2.isye.gatech.edu")||(Machine=="isye-jacobi3.isye.gatech.edu")||(Machine=="isye-jacobi4.isye.gatech.edu")||(Machine=="isye-jacobi5.isye.gatech.edu")||(Machine=="isye-jacobi1.isye.gatech.edu")||(Machine=="isye-leibniz2.isye.gatech.edu")||(Machine=="isye-leibniz3.isye.gatech.edu")||(Machine=="isye-leibniz4.isye.gatech.edu")||(Machine=="isye-leibniz5.isye.gatech.edu")||(Machine=="isye-jacobi7.isye.gatech.edu")||(Machine=="isye-jacobi4.isye.gatech.edu"))')
lines.append('arguments=$ENV(HOME)/BoRisk/cluster_main.py' + args)
lines.append('log=run' + args_ + '.log')
lines.append('error=run' + args_ + '.error')
lines.append('output=run' +args_ + '.out')
lines.append('notification=error')
lines.append('notification=complete')
lines.append('notify_user=scakmak3@gatech.edu')
lines.append('request_memory = 8196')
lines.append('queue')

for line in lines:
    file.write(line)
    file.write('\n')
file.close()
