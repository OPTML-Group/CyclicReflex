import os
import random
import shutil
import time

def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    """
    负责分配任务并生成对应 GPU 的执行脚本
    """
    if len(commands) == 0:
        return 
    if os.path.exists(dir):
        shutil.rmtree(dir)
    
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
        
    os.makedirs(dir, exist_ok=True)

    # 生成停止脚本
    fout = open('stop_{}.sh'.format(dir), 'w')
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if len(i_commands) == 0:
            continue 

        sh_path = os.path.join(dir, "run{}.sh".format(i))
        fout = open(sh_path, 'w')
        for com in i_commands:
            print(com, file=fout)
        fout.close()
        
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)


def gen_commands_search():
    """
    生成超参数搜索的命令行
    """
    # 假设你有 8 张卡，如果不是请修改这个列表，例如 [0, 1, 2, 3]
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    gpu_index = 0  

    commands_by_gpu = {gpu: [] for gpu in gpus}

    # 基础配置
    wandb_project = "cylicreflex"
    seed = 42
    mode = "CoT_withThinking"
    logits_processor_type = "cyclical"
    
    # 固定的超参数
    wait_cyclical_amplitude = 5.0 # 5.0 -> 3.0
    wait_cyclical_shift = 0

    # models = ["qwen1.5b", "qwen7b", "llama8b"]
    # datasets = ["math500", "amc2023", "aime2024", "aime2025"]

    # for model in models:
    #     for dataset in datasets:
            
    #         # 动态分配 period 搜索范围
    #         if dataset in ["math500", "amc2023"]:
    #             # period_list = [500, 1000, 1500]
    #             # period_list = [400, 600, 800, 1200]
    #             period_list = [300, 700, 900, 1100, 1300]
    #         elif dataset in ["aime2024", "aime2025"]:
    #             # period_list = [1000, 2000, 3000]
    #             period_list = [1200, 1400, 1600, 1800]
    #         else:
    #             continue

    # models = ["llama8b"]
    # datasets = ["amc2023", "aime2024", "aime2025"]

    # for model in models:
    #     for dataset in datasets:

    #         # 动态分配 period 搜索范围
    #         if dataset in ["math500", "amc2023"]:
    #             period_list = [300, 700, 900, 1100, 1300]
    #         elif dataset in ["aime2024", "aime2025"]:
    #             period_list = [800, 900, 1100, 1300, 1500]
    #         else:
    #             continue

    models = ["qwen7b"]
    datasets = ["amc2023"]

    for model in models:
        for dataset in datasets:

            # 动态分配 period 搜索范围
            if dataset in ["math500", "amc2023"]:
                period_list = [300, 700, 900, 1100, 1300]
            elif dataset in ["aime2024", "aime2025"]:
                period_list = [800, 900, 1100, 1300, 1500]
            else:
                continue

            for wait_cyclical_period in period_list:
                current_gpu = gpus[gpu_index]

                # 组装命令行
                command = (
                    f"CUDA_VISIBLE_DEVICES={current_gpu} python3 evaluate.py "
                    f"--model_choice {model} "
                    f"--datasets {dataset} "
                    f"--max_new_tokens 8192 "
                    f"--wandb_project {wandb_project} "
                    f"--logits_processor_type {logits_processor_type} "
                    f"--wait_cyclical_amplitude {wait_cyclical_amplitude} "
                    f"--wait_cyclical_period {wait_cyclical_period} "
                    f"--wait_cyclical_shift {wait_cyclical_shift} "
                    f"--mode {mode} "
                    f"--seed {seed}"
                )
                
                commands_by_gpu[current_gpu].append(command)
                gpu_index = (gpu_index + 1) % len(gpus)

    # 均匀交错合并各 GPU 的命令，确保负载均衡
    flat_commands = []
    max_length = max(len(cmds) for cmds in commands_by_gpu.values())
    for i in range(max_length):
        for gpu in gpus:
            if i < len(commands_by_gpu[gpu]):
                flat_commands.append(commands_by_gpu[gpu][i])
                
    return flat_commands


if __name__ == "__main__":
    
    # 生成命令列表
    commands = gen_commands_search()
    print(f"[*] 共生成了 {len(commands)} 条测试命令。")
    
    # 请根据你实际可用的 GPU 编号修改这里的列表
    available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # 执行命令分发 (shuffle=True 让不同实验随机打乱，避免长耗时的实验扎堆)
    run_commands(
        gpus=available_gpus, 
        commands=commands, 
        call=True,  # 设为 True 则直接在后台跑，设为 False 则只生成 sh 脚本
        dir="commands_final", 
        shuffle=False, 
        delay=0.5
    )