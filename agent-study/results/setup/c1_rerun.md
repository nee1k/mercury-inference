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

```text
$ date -u +%FT%TZ
2026-11-27T06:41:32Z
```

```text
$ nvidia-smi -i 2,3 --query-gpu=index,name,memory.total,driver_version,power.limit,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current --format=csv
index, name, memory.total [MiB], driver_version, power.limit [W], pcie.link.gen.max, pcie.link.gen.current, pcie.link.width.max, pcie.link.width.current
2, NVIDIA H200 NVL, 143771 MiB, 610.43.02, 600.00 W, 5, 5, 16, 16
3, NVIDIA H200 NVL, 143771 MiB, 610.43.02, 600.00 W, 5, 5, 16, 16
```

```text
$ python -B c1_probe.py live 2
vllm:cache_config_info{_block_size_resolved="True",block_size="16",cache_dtype="auto",calculate_kv_scales="False",enable_prefix_caching="True",engine="0",gpu_memory_utilization="0.9",hash_block_size="None",is_attention_free="False",kv_cache_dtype_skip_layers="[]",kv_cache_memory_bytes="None",kv_offloading_backend="native",kv_offloading_size="None",kv_sharing_fast_prefill="False",mamba_block_size="None",mamba_cache_dtype="auto",mamba_cache_mode="none",mamba_page_size_padded="None",mamba_ssm_cache_dtype="auto",num_cpu_blocks="None",num_gpu_blocks="15694",num_gpu_blocks_override="None",prefix_caching_hash_algo="sha256",sliding_window="None",user_specified_block_size="False",user_specified_mamba_block_size="False"} 1.0
gpu=2 prompt_tokens=1024 trial=1 ttft_s=0.121537267 cached_tokens=0 usage_cached_tokens=null prefix_hits_delta=0 prefix_queries_delta=1024
gpu=2 prompt_tokens=1024 trial=2 ttft_s=0.115322950 cached_tokens=0 usage_cached_tokens=null prefix_hits_delta=0 prefix_queries_delta=1024
gpu=2 prompt_tokens=1024 trial=3 ttft_s=0.114528194 cached_tokens=0 usage_cached_tokens=null prefix_hits_delta=0 prefix_queries_delta=1024
gpu=2 prompt_tokens=4096 trial=1 ttft_s=0.469715669 cached_tokens=0 usage_cached_tokens=null prefix_hits_delta=0 prefix_queries_delta=4096
gpu=2 prompt_tokens=4096 trial=2 ttft_s=0.474046268 cached_tokens=0 usage_cached_tokens=null prefix_hits_delta=0 prefix_queries_delta=4096
gpu=2 prompt_tokens=4096 trial=3 ttft_s=0.469237120 cached_tokens=0 usage_cached_tokens=null prefix_hits_delta=0 prefix_queries_delta=4096
gpu=2 prompt_tokens=16384 trial=1 ttft_s=2.267046604 cached_tokens=0 usage_cached_tokens=null prefix_hits_delta=0 prefix_queries_delta=16384
gpu=2 prompt_tokens=16384 trial=2 ttft_s=2.265548215 cached_tokens=0 usage_cached_tokens=null prefix_hits_delta=0 prefix_queries_delta=16384
gpu=2 prompt_tokens=16384 trial=3 ttft_s=2.274423853 cached_tokens=0 usage_cached_tokens=null prefix_hits_delta=0 prefix_queries_delta=16384
```

```text
$ python -B c1_probe.py copies 2
gpu=2 direction=H2D trial=1 bytes=1073741824 milliseconds=19.042368
gpu=2 direction=H2D trial=2 bytes=1073741824 milliseconds=19.004416
gpu=2 direction=H2D trial=3 bytes=1073741824 milliseconds=18.962591
gpu=2 direction=H2D trial=4 bytes=1073741824 milliseconds=18.962145
gpu=2 direction=H2D trial=5 bytes=1073741824 milliseconds=18.960257
gpu=2 direction=H2D trial=6 bytes=1073741824 milliseconds=18.962879
gpu=2 direction=H2D trial=7 bytes=1073741824 milliseconds=18.956064
gpu=2 direction=H2D trial=8 bytes=1073741824 milliseconds=18.960287
gpu=2 direction=H2D trial=9 bytes=1073741824 milliseconds=18.953953
gpu=2 direction=H2D trial=10 bytes=1073741824 milliseconds=18.946016
gpu=2 direction=H2D median_GB_per_s=56.628321
gpu=2 direction=D2H trial=1 bytes=1073741824 milliseconds=19.130655
gpu=2 direction=D2H trial=2 bytes=1073741824 milliseconds=19.128927
gpu=2 direction=D2H trial=3 bytes=1073741824 milliseconds=19.135616
gpu=2 direction=D2H trial=4 bytes=1073741824 milliseconds=19.130016
gpu=2 direction=D2H trial=5 bytes=1073741824 milliseconds=19.123455
gpu=2 direction=D2H trial=6 bytes=1073741824 milliseconds=19.126335
gpu=2 direction=D2H trial=7 bytes=1073741824 milliseconds=19.112736
gpu=2 direction=D2H trial=8 bytes=1073741824 milliseconds=19.114784
gpu=2 direction=D2H trial=9 bytes=1073741824 milliseconds=19.130304
gpu=2 direction=D2H trial=10 bytes=1073741824 milliseconds=19.109535
gpu=2 direction=D2H median_GB_per_s=56.135641
```

```text
$ python -B c1_probe.py copies 3
gpu=3 direction=H2D trial=1 bytes=1073741824 milliseconds=19.037472
gpu=3 direction=H2D trial=2 bytes=1073741824 milliseconds=18.960768
gpu=3 direction=H2D trial=3 bytes=1073741824 milliseconds=18.958879
gpu=3 direction=H2D trial=4 bytes=1073741824 milliseconds=18.960320
gpu=3 direction=H2D trial=5 bytes=1073741824 milliseconds=18.962240
gpu=3 direction=H2D trial=6 bytes=1073741824 milliseconds=18.956352
gpu=3 direction=H2D trial=7 bytes=1073741824 milliseconds=18.959871
gpu=3 direction=H2D trial=8 bytes=1073741824 milliseconds=18.948128
gpu=3 direction=H2D trial=9 bytes=1073741824 milliseconds=18.955936
gpu=3 direction=H2D trial=10 bytes=1073741824 milliseconds=18.960287
gpu=3 direction=H2D median_GB_per_s=56.631716
gpu=3 direction=D2H trial=1 bytes=1073741824 milliseconds=19.145920
gpu=3 direction=D2H trial=2 bytes=1073741824 milliseconds=19.138752
gpu=3 direction=D2H trial=3 bytes=1073741824 milliseconds=19.141121
gpu=3 direction=D2H trial=4 bytes=1073741824 milliseconds=19.127329
gpu=3 direction=D2H trial=5 bytes=1073741824 milliseconds=19.134336
gpu=3 direction=D2H trial=6 bytes=1073741824 milliseconds=19.136703
gpu=3 direction=D2H trial=7 bytes=1073741824 milliseconds=19.129856
gpu=3 direction=D2H trial=8 bytes=1073741824 milliseconds=19.132065
gpu=3 direction=D2H trial=9 bytes=1073741824 milliseconds=19.139969
gpu=3 direction=D2H trial=10 bytes=1073741824 milliseconds=19.132193
gpu=3 direction=D2H median_GB_per_s=56.112498
```
