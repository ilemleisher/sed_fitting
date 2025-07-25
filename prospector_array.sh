#! /bin/bash
#SBATCH --job-name=__ #000priors0 for simba, 000priors0_smgl for smuggle
#SBATCH --output=%x.o
#SBATCH --error=%x.e
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leisheri@ufl.edu
#SBATCH --time=48:00:00
#SBATCH --tasks-per-node=2
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=3800
#SBATCH --account=narayanan
#SBATCH --qos=__ #delete -b for investment
#SBATCH --array=__

JOBNAME=$SLURM_JOB_NAME
SNAPSHOT=${JOBNAME:0:3}

if [[ ${#JOBNAME} -eq 10 ]]; then
    PRIORS=${JOBNAME:3}
    ID=$(awk '{if(NR==(n+1)) print int($0)}' n=${SLURM_ARRAY_TASK_ID} /orange/narayanan/leisheri/simba/m25n512/snap${SNAPSHOT}/snap${SNAPSHOT}_gas_gals.txt)
    GALAXIES="caesar"
    
elif [[ ${#JOBNAME} -eq 15 ]]; then
    PRIORS=${JOBNAME:3:$(( ${#JOBNAME}-8 ))}
    ID=${SLURM_ARRAY_TASK_ID}
    GALAXIES="subfind"
fi

cd /home/leisheri
module purge
module load conda
conda activate /blue/narayanan/leisheri/conda/envs/env5
cd /home/leisheri/many_jobs/

python run_prosp.py ${SNAPSHOT} ${ID} ${PRIORS} ${GALAXIES} 
python prospector_script.py ${SNAPSHOT} ${ID} ${PRIORS} ${GALAXIES}
