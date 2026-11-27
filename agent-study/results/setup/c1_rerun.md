# C1 rerun setup and measurements

Commands run from agent-study with the study Python and LC_ALL=C.

```text
$ date -u +%FT%TZ
2026-11-27T02:40:39Z
```

```text
$ nvidia-smi -i 2,3 --query-gpu=index,name,memory.total,driver_version,power.limit,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current --format=csv
index, name, memory.total [MiB], driver_version, power.limit [W], pcie.link.gen.max, pcie.link.gen.current, pcie.link.width.max, pcie.link.width.current
2, NVIDIA H200 NVL, 143771 MiB, 610.43.02, 600.00 W, 5, 5, 16, 16
3, NVIDIA H200 NVL, 143771 MiB, 610.43.02, 600.00 W, 5, 5, 16, 16
```

```text
$ nvidia-smi | grep -E 'CUDA (UMD )?Version'
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
```

```text
$ grep '^MemTotal:' /proc/meminfo
MemTotal:       1187958768 kB
```

```text
$ lscpu | grep -E '^(CPU\(s\):|Model name:|Core\(s\) per socket:|Socket\(s\):)'
CPU(s):                                  128
Model name:                              AMD EPYC 9555 64-Core Processor
Core(s) per socket:                      64
Socket(s):                               2
```

```text
$ python -B c1_probe.py inventory
Python 3.12.13
torch 2.11.0+cu130
vllm 0.21.0
transformers 5.6.0
snapshot 9216db5781bf21249d130ec9da846c4624c16137
num_hidden_layers 64
hidden_size 5120
num_attention_heads 64
num_key_value_heads 8
head_dim 128
intermediate_size 25600
vocab_size 151936
tie_word_embeddings false
```
