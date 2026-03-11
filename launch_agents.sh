num_agents=6

for i in $(seq 1 $num_agents); do
    wandb agent nihermann/param-tuning-demo/80slmdqz &
done