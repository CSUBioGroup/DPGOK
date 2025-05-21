#!/bin/bash
#SBATCH -A pi_test              ## 指定用户个人计费账户,GPU资源需缴费使用
#SBATCH -p gpu2Q                ## 指定提交作业的分区
#SBATCH -q gpuq                 ## 指定qos
#SBATCH -o job.%j.out       ## 指定作业标准输出文件
#SBATCH -J split_dataset         ## 指定作业名
#SBATCH --nodes=1               ## 指定节点数
#SBATCH --ntasks-per-node=1     ## 指定每节点任务数，也就是并行的进程数
#SBATCH --gres=gpu:1            ## 指定需要使用的GPU卡数

## 用户若使用了其他版本的conda，则下方命令对应的路径也许更改
source /public/software/anaconda2022/etc/profile.d/conda.sh
conda activate deepgose

python /public/home/hpc244706074/myProject/utils/split_dataset.py --uniprot_sport_file1 /public/home/hpc244706074/myProject/data/uniprot_202201/uniprot_sprot.dat --uniprot_sport_file2 /public/home/hpc244706074/myProject/data/uniprot_202301/uniprot_sprot.dat --uniprot_sport_file3 /public/home/hpc244706074/myProject/data/uniprot_202405/uniprot_sprot.dat --output_file /public/home/hpc244706074/myProject/data --go_file /public/home/hpc244706074/myProject/data/go.obo --bp_freq 1 --cc_freq 1 --mf_freq 1