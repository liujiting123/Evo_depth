# VLA-Arena Evaluation
## Environment Setup
For environment setup, you can follow the instructions below:
``` bash
cd VLA_Arena_Evaluation
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena
conda create -n vla_arena python=3.11
pip install .
pip install websockets==15.0.1 draccus
cd ..


```

## Evaluation 
In one terminal, you can run the server scripts.

``` bash
cd Evo_depth
python scripts/Evo1_server.py
```

In the other terminal, you can run the evaluation scripts.

``` bash
cd VLA_Arena_Evaluation
python vla_arena/vla_arena_client.py  \
--execution_horizon 10     \
--seed 10    \
--num_episodes_per_task 10  \  
--server_url ws://127.0.0.1:9000  \  
--log_out_dir ./logs/exp_h10_s27  
--save_video_mode all \
--max_episode_steps 300 
```