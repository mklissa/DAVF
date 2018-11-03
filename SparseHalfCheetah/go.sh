#!/usr/bin/bash

seed_start=0

if [ $1 == "0" ]; then
        name="no_graph"
else
        name="yes_graph"
fi


tmux new-session -d -s $name


for ((i=0;i<10;i++));do

        let seed=seed_start+i

        tmux new-window -t $name: -n "seed$seed"
        tmux send-keys -t $name: "bash"
        tmux send-keys -t $name:  Enter
        tmux send-keys -t $name: "source activate soc"
        tmux send-keys -t $name:  Enter
        tmux send-keys -t $name: "python run_mujoco.py --seed $seed --orig_gen $1"
        tmux send-keys -t $name:  Enter

done

tmux -2 attach-session -t $name