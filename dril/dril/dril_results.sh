# first install and generate data following the readme
# then below

mkdir -p outputs
ID=1001num_ensemble_train_epoch_1001bc_train_epoch_1000num_updates_online # 1000 num updates is 1M steps atleast
mkdir -p outputs/dril_${ID}

############## atari-pong
G=0
for N in 7 10 13 16
do
    CUDA_VISIBLE_DEVICES=${G} python -u main.py --env-name PongNoFrameskip-v4 --default_experiment_params atari \
    --num-trajs ${N} --rl_baseline_zoo_dir /home/ksridhar/ICDM/dril/rl-baselines-zoo --seed 0  --dril \
    &> outputs/dril_${ID}/atari-pong_${N}demos.txt &

    G=$(((G+1)%10))
done

############### atari-mspacman
for N in 4 5 9 10
do
    CUDA_VISIBLE_DEVICES=${G} python -u main.py --env-name MsPacmanNoFrameskip-v4 --default_experiment_params atari \
    --num-trajs ${N} --rl_baseline_zoo_dir /home/ksridhar/ICDM/dril/rl-baselines-zoo --seed 0  --dril \
    &> outputs/dril_${ID}/atari-mspacman_${N}demos.txt &

    G=$(((G+1)%10))
done

############### atari-alien
for N in 3 4 5 6
do
    CUDA_VISIBLE_DEVICES=${G} python -u main.py --env-name AlienNoFrameskip-v4 --default_experiment_params atari \
    --num-trajs ${N} --rl_baseline_zoo_dir /home/ksridhar/ICDM/dril/rl-baselines-zoo --seed 0  --dril \
    &> outputs/dril_${ID}/atari-alien_${N}demos.txt &

    G=$(((G+1)%10))
done