CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 /home/wangzb/challenges/lehome-challenge/.venv/bin/lerobot-train --config_path=configs/train_pi05.yaml


pi05的transformer需要lerobot之外单独安装
GIT_LFS_SKIP_SMUDGE=1 uv pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git@v0.4.3"


GPU0功率异常高，但是什么也没跑，怀疑down了，导致不能多卡跑

需要使用base里的ffmpeg库，uv里没有，系统也没有
export LD_LIBRARY_PATH="$HOME/miniconda3/lib:$LD_LIBRARY_PATH"

merged下载到本地总是和hugging face的大小不一样
然后还有报错ValueError: No valid stream found in input file. Is -1 of the desired media type?
我还以为是数据错了，结果是ffmpeg版本问题

应该用这个
conda install -c conda-forge ffmpeg=6.1.1 -y

md显存不够，为啥之前看batch8微调还行，可能数据集占地方了

换H200


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 /home/wangzb/challenges/lehome-challenge/.venv/bin/lerobot-train --config_path=configs/train_pi05.yaml

128爆显存了 64也爆了 32也爆了 最后单卡16一共32了


CUDA_VISIBLE_DEVICES=1 python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_h200_new/checkpoints/300000/pretrained_model \
    --dataset_root Datasets/example/top_long_merged \
    --garment_type "top_long" \
    --num_episodes 5 \
    --enable_cameras \
    --headless \
    --save_video \
    --video_dir outputs/eval_videos/act_h200_300k
    --device cpu

这个可以直接在下载的目录测
CUDA_VISIBLE_DEVICES=2 python -m scripts.eval --policy_type custom --garment_type top_long --enable_cameras --device cpu


L40是真不方便，测试也不行md
一堆红，而且很多东西我还下载不了，换成H200了，一点错没有
开了虚拟的显示
Xvfb :1 -screen 0 1280x720x24 &
export DISPLAY=:1

export __GLX_VENDOR_LIBRARY_NAME=nvidia

[nvidia-ngx-upda] <defunct>
碍事吗，一直启动不了，进行不下去


这个也依旧卡死
xvfb-run -a python -m scripts.eval --policy_type custom --garment_type top_long --enable_cameras --device cpu

哦md是不是H200不能用RTX啊



CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 /home/wzb/challenges/lehome-challenge/.venv/bin/lerobot-train --config_path=/home/wzb/challenges/lehome-challenge/outputs/train/pi05_merged/checkpoints/last/pretrained_model/train_config.json --resume=true  --steps=50000



CUDA_VISIBLE_DEVICES=1 python -m scripts.eval \
  --headless \
  --policy_type custom \
  --garment_type top_long \
  --enable_cameras \
  --device cpu


xvfb-run -a python -m scripts.eval \
  --headless \
  --policy_type custom \
  --garment_type top_long \
  --enable_cameras \
  --device cpu


CUDA_VISIBLE_DEVICES=1 

__NV_DISABLE_NGX_UPDATER=1 

CUDA_VISIBLE_DEVICES=1  xvfb-run -a .venv/bin/python -m scripts.eval \
    --policy_type lerobot \
    --policy_path /home/wzb/challenges/weights/act_moe/outputs/train/act_h200_top_long/checkpoints/last/pretrained_model \
    --dataset_root /data/datasets/datasets-hf/dataset_challenge_merged/top_long_merged \
    --garment_type "top_long" \
    --num_episodes 5 \
    --enable_cameras \
    --headless \
    --save_video \
    --video_dir outputs/eval_videos/act_h200_top_long \
    --device cpu



CUDA_VISIBLE_DEVICES=1 xvfb-run -a .venv/bin/python -m scripts.eval \
    --policy_type lerobot \
    --policy_path /home/wzb/challenges/lehome-challenge/outputs/train/pi05_merged_fine/checkpoints/last/pretrained_model \
    --dataset_root /data/datasets/datasets-hf/dataset_challenge_merged/top_long_merged \
    --garment_type "top_long" \
    --num_episodes 5 \
    --enable_cameras \
    --headless \
    --save_video \
    --video_dir outputs/eval_videos/pi05_fine_h200_top_long \
    --device cpu



CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 /home/wzb/challenges/lehome-challenge/.venv/bin/lerobot-train --config_path=/home/wzb/challenges/lehome-challenge/outputs/train/pi05_merged/checkpoints/last/pretrained_model/train_config.json --resume=true  --steps=50000  --policy.optimizer_lr=2e-5  --policy.scheduler_decay_lr=1e-5  --scheduler.decay_lr=1e-5 --scheduler.peak_lr=2e-5  --optimizer.lr=2e-5


CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 /home/wzb/challenges/lehome-challenge/.venv/bin/lerobot-train --config_path=/home/wzb/challenges/lehome-challenge/outputs/train/pi05_merged/checkpoints/nostate/pretrained_model/train_config.json --resume=true  --steps=50000  --policy.optimizer_lr=2e-5  --policy.scheduler_decay_lr=1e-5  --scheduler.decay_lr=1e-5 --scheduler.peak_lr=2e-5  --optimizer.lr=2e-5

删除training state报错，算了重新设置来吧

中途想要改学习率的话，只改policy的还不行，schduler也要改
为什么这里的optimizer也有lr，看不懂，不是scheduler做的吗




CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 /home/wzb/challenges/lehome-challenge/.venv/bin/lerobot-train --config_path=configs/fine_pi05.yaml





CUDA_VISIBLE_DEVICES=0 xvfb-run -a .venv/bin/python -m scripts.eval \
    --policy_type lerobot \
    --policy_path /home/wzb/challenges/lehome-challenge/outputs/train/pi05_merged_fine/checkpoints/last/pretrained_model \
    --dataset_root /data/datasets/datasets-hf/dataset_challenge_merged/pant_long_merged \
    --garment_type "pant_long" \
    --num_episodes 5 \
    --enable_cameras \
    --headless \
    --save_video \
    --video_dir outputs/eval_videos/pi05_fine_h200_pant_long \
    --device cpu



CUDA_VISIBLE_DEVICES=1 xvfb-run -a .venv/bin/python -m scripts.eval \
    --policy_type lerobot \
    --policy_path /home/wzb/challenges/lehome-challenge/outputs/train/pi05_merged_fine/checkpoints/last/pretrained_model \
    --dataset_root /data/datasets/datasets-hf/dataset_challenge_merged/pant_short_merged \
    --garment_type "pant_short" \
    --num_episodes 5 \
    --enable_cameras \
    --headless \
    --save_video \
    --video_dir outputs/eval_videos/pi05_fine_h200_pant_short \
    --device cpu



pi05二次训练后的结果

chunksize=10，n_action_steps=10
|task|seen|unseen|all|
|:--:|:--:|:--:|:--:|
|top_long|62%|70%|63.3%|
|top_short|20%|0%|16.7%|
|pant_long|14%|0%|11.7%|
|pant_short|94%|50%|86.6%|



chunksize=10，n_action_steps=5
|task|seen|unseen|all|
|:--:|:--:|:--:|:--:|
|top_long|62%|70%|63.3%|
|top_short|20%|0%|16.7%|
|pant_long|0.0|0.0||
|pant_short|94%|50%|86.6%|












CUDA_VISIBLE_DEVICES=1 xvfb-run -a .venv/bin/python -m scripts.eval \
    --policy_type lerobot \
    --policy_path /home/wzb/challenges/lehome-challenge/outputs/train/pi05_merged_fine/checkpoints/last/pretrained_model \
    --dataset_root /data/datasets/datasets-hf/dataset_challenge_merged/top_short_merged \
    --garment_type "top_short" \
    --num_episodes 1 \
    --enable_cameras \
    --headless \
    --save_video \
    --video_dir outputs/eval_videos/test \
    --device cpu



数据集映射可能出了问题, parquet记录的内容index和garment json的顺序可能不一致

md好像是我写错了，我看eval时seen 3是11381但是我自己打印的没有这么多是10869
哦还有一个device的问题，貌似是cpu vs cuda
原路径eval用的cpu，我好像用的cuda



xvfb-run -a .venv/bin/python -m scripts.dataset_sim filter \
  --dataset_root /data/datasets/datasets-hf/dataset_challenge_merged/top_short_merged \
  --seen_indices 3 \
  --start_episode 75 \
  --end_episode 80 \
  --report_root outputs/filter_reports_top_short_seen3_cpu_small \
  --output_root /data/datasets/datasets-hf/dataset_challenge_filteredtest_small \
  --headless \
  --enable_cameras \
  --step_hz 0 \
  --device cpu






CUDA_VISIBLE_DEVICES=0 xvfb-run -a .venv/bin/python -m scripts.dataset_sim filter \
  --dataset_root /data/datasets/datasets-hf/dataset_challenge_merged/top_short_merged \
  --report_root outputs/filter_reports_top_short \
  --output_root /data/datasets/datasets-hf/dataset_challenge_filtered/top_short \
  --headless \
  --enable_cameras \
  --step_hz 0 \
  --device cpu


CUDA_VISIBLE_DEVICES=1 xvfb-run -a .venv/bin/python -m scripts.dataset_sim filter \
  --dataset_root /data/datasets/datasets-hf/dataset_challenge_merged/pant_long_merged \
  --report_root outputs/filter_reports_pant_long \
  --output_root /data/datasets/datasets-hf/dataset_challenge_filtered/pant_long \
  --headless \
  --enable_cameras \
  --step_hz 0 \
  --device cpu




CUDA_VISIBLE_DEVICES=0 xvfb-run -a .venv/bin/python -m scripts.dataset_sim filter \
  --dataset_root /data/datasets/datasets-hf/dataset_challenge_merged/top_long_merged \
  --report_root outputs/filter_reports_top_long \
  --output_root /data/datasets/datasets-hf/dataset_challenge_filtered/top_long \
  --headless \
  --enable_cameras \
  --step_hz 0 \
  --device cpu


CUDA_VISIBLE_DEVICES=1 xvfb-run -a .venv/bin/python -m scripts.dataset_sim filter \
  --dataset_root /data/datasets/datasets-hf/dataset_challenge_merged/pant_short_merged \
  --report_root outputs/filter_reports_pant_short \
  --output_root /data/datasets/datasets-hf/dataset_challenge_filtered/pant_short \
  --headless \
  --enable_cameras \
  --step_hz 0 \
  --device cpu





pant_short 244/250 这个效果在我旧版测试最好
pant_long 164/250 这个测试就很垃圾

