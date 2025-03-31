#!/bin/bash
# beta(关闭gpu开关,--use_gpu 0):
#vc-test submit --sync --image docker.v2.aispeech.com/hpc-base/ai_base-aps-fireredasr:FireRedASR-v0.0.1-004 --cpu-per-task 32 --mem-per-task 48G --partition pdcpu-test --num-task 2 --project ai-core-aps-FireRedASR --dir /opt/FireRedASR --job FireRedASR-Main-Job --type pytorch --cmd "/opt/FireRedASR/examples/run.sh --output /mnt/lustre/hpc_stor01/home/lang.zhang/data/tmp/FireRedASR/output --asr_type aed --wav_scp /mnt/lustre/hpc_stor01/home/lang.zhang/data/angji_wav/a.scp --use_gpu 0"

# 设置默认参数值
asr_type="aed"
wav_scp="/opt/FireRedASR/fireredasr/examples/wav.scp"
output=""  # 将output默认设为空，强制用户必须传入
use_gpu="1"
batch_size="2"
beam_size="3"

# 处理传入参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --asr_type)
            asr_type="$2"
            shift 2
            ;;
        --wav_scp)
            wav_scp="$2"
            shift 2
            ;;
        --output)
            output="$2"
            shift 2
            ;;
        --use_gpu)
            use_gpu="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --beam_size)
            beam_size="$2"
            shift 2
            ;;
        *)
            echo "错误: 未知参数 '$1'"
            echo "允许的参数: --asr_type, --wav_scp, --output, --decode_args"
            exit 1
            ;;
    esac
done

decode_args="--batch_size $batch_size --beam_size $beam_size --nbest 1 --decode_max_len 0 --softmax_smoothing 1.25 --aed_length_penalty 0.6 --eos_penalty 1.0 --use_gpu $use_gpu"

# 检查output参数是否为空
if [ -z "$output" ]; then
    echo "错误: output参数为必传字段且不能为空！"
    echo "请使用 --output 参数指定输出目录"
    exit 1
fi

# 打印使用的参数值（调试用）
echo "使用的参数:"
echo "asr_type: $asr_type"
echo "wav_scp: $wav_scp"
echo "output: $output"
echo "decode_args: $decode_args"

# 设置模型目录
model_dir='/hpc_stor01/project/ezkws/ai-aps/fireredasr/huggingface/models/FireRedASR-AED-L'

# 创建输出目录（如果不存在）
mkdir -p "$output"

# 读取环境变量
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"

# 当 WORLD_SIZE > 1 时，RANK=0 负责拆分数据
data_dir="$output/data_split"
if [ "$WORLD_SIZE" -gt 1 ]; then
    mkdir -p "$data_dir"
    if [ "$RANK" -eq 0 ]; then
        echo "开始拆分数据..."
        total_lines=$(wc -l < "$wav_scp")
        lines_per_proc=$(( (total_lines + WORLD_SIZE - 1) / WORLD_SIZE ))
        # split -l $lines_per_proc "$wav_scp" "$data_dir/part_"
        split -l $lines_per_proc -d -a 2 "$wav_scp" "$data_dir/part_"  # 使用数字后缀
        echo "数据拆分完成"
    fi
    
    # 等待RANK=0完成数据拆分，检查所有part_*文件是否生成
    while [ $(ls "$data_dir" | grep -c "part_..$") -lt $((WORLD_SIZE)) ]; do
        sleep 1
    done
    
    part_file=$(printf "%s/part_%s" "$data_dir" $(printf "%02d" $RANK))
else
    part_file="$wav_scp"
fi

# 每个进程处理自己的数据
if [ -f "$part_file" ]; then
    echo "进程 $RANK 开始处理 $part_file..."
    export PATH=/opt/FireRedASR/fireredasr/:/opt/FireRedASR/fireredasr/utils/:$PATH
    export PYTHONPATH=/opt/FireRedASR/:$PYTHONPATH
    cd /opt/FireRedASR/examples
    speech2text.py --asr_type "$asr_type" --model_dir "$model_dir" $decode_args --wav_scp "$part_file" --output "$output/asr_$RANK.txt"

    if [ $? -eq 0 ]; then
        echo "进程 $RANK 处理完成！结果保存在 $output/asr_$RANK.txt"
        touch "$output/.done-$RANK"
    else
        echo "错误: 进程 $RANK 处理失败！"
        exit 1
    fi
else
    echo "进程 $RANK 未找到对应的数据文件: $part_file"
    exit 1
fi

# RANK=0 合并所有结果（仅在 WORLD_SIZE > 1 时需要等待）
if [ "$RANK" -eq 0 ]; then
    if [ "$WORLD_SIZE" -gt 1 ]; then
        echo "等待所有进程完成..."
        for (( i=0; i<$WORLD_SIZE; i++ )); do
            while [ ! -f "$output/.done-$i" ]; do
                sleep 1
            done
        done
    fi
    echo "开始合并结果..."
    cat "$output"/asr_*.txt > "$output/asr.txt"
    echo "所有进程完成，最终结果保存在 $output/asr.txt"
fi
