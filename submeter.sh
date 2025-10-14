#!/bin/bash
#SBATCH --nodes=1                      #Numero de Nós deve ser igual a 1
#SBATCH --ntasks-per-node=1            #Numero de tarefas por Nó deve ser igual a 1
#SBATCH --ntasks=1                     #Numero total de tarefas deve ser igual a 1
#SBATCH --cpus-per-task=192            #Numero de threads
#SBATCH -p ict-gh200                   #Fila (partition) a ser utilizada
#SBATCH -J Hopscotch                   #Nome job
#SBATCH --time=00:30:00		       #Altera o tempo limite para 1 hora
#SBATCH --account=superpd
#SBATCH --gpus=1



#Exibe os nós alocados para o Job
#echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIR

#Configura os compiladores
#module load intel_psxe/2020_sequana

#exibe informações sobre o executável
EXEC=/petrobr/parceirosbr/home/frederico.cabral2/HOPSCOTCH/hopscotch2d_omp_semaphores
#/usr/bin/ldd  $EXEC

#Configura o numero de threads OpenMP com o valor passado para a variavel --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -c $SLURM_CPUS_PER_TASK $EXEC

