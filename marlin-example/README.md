# nm-vllm Acceleration with Quantization

This this example, we will demonstrate the inference acceleration from quantization using the Marlin kernels for INT4 infernence.

> Note: this example requires Ampere GPUs or later.

## Spin Up a Server Running a Marlin Model

We can use `neuralmagic/zephyr-7b-beta-marlin`, which is posted on the Hugging Face hub.

Download the docker image to get started:

```bash
docker pull ghcr.io/neuralmagic/nm-vllm-openai:v0.1.0
docker tag ghcr.io/neuralmagic/nm-vllm-openai:v0.1.0 nm-vllm:v0.1.0
```

### Deploy INT4 Model

Launch:

```bash
docker run \
    --gpus all \
    --shm-size 2g \
    -p 8000:8000 \
    nm-vllm:v0.1.0 --model neuralmagic/zephyr-7b-beta-marlin --max-model-len 4096 --disable-log-requests
```

### Benchmark the INT4 Model

When evaluating LLM performance, there are two latency metrics to consider. 
- TTFT (Time to first token) measures how long it takes to generate the first token. 
- TPOT (Time per output token) measures how long it takes to generate each incremental token.

The benchmark scripts provided here help us to evaluate these metrics.

#### Install
Install dependencies to run the benchmark client.

```bash
python3 -m venv benchmark-env
source benchmark-env/bin/activate
pip install -U aiohttp transformers
```

#### Run the Benchmark

Download some sample data:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Launch the benchmark script. We run 1 query per second.

```bash
python3 benchmark_serving.py \
    --backend openai \
    --endpoint /v1/completions \
    --model neuralmagic/zephyr-7b-beta-marlin \
    --request-rate 1.0 \
    --num-prompts 100 \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json 
```

We can see that the quantized models gets `12ms` TPOT.

```bash
---------------Time to First Token----------------
Mean TTFT (ms):                          68.66     
Median TTFT (ms):                        28.75     
P99 TTFT (ms):                           250.87    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          12.04     
Median TPOT (ms):                        11.84     
P99 TPOT (ms):                           21.41     
==================================================
```

### Deploy FP16 Model

Now, let's deploy the FP16 model.

```bash
docker run \
    --gpus all \
    --shm-size 2g \
    -p 8000:8000 \
    nm-vllm:v0.1.0 --model HuggingFaceH4/zephyr-7b-beta --max-model-len 4096 --disable-log-requests
```

### Benchmark the FP16 Model

We can use the same script, just swapping out the model. We run 1 query per second.

```bash
python3 benchmark_serving.py \
    --backend openai \
    --endpoint /v1/completions \
    --model HuggingFaceH4/zephyr-7b-beta \
    --request-rate 1.0 \
    --num-prompts 100 \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json 
```

We can see the fp16 models get `39ms` TPOT, about 3.5x slower than the INT4 model.

```bash
---------------Time to First Token----------------
Mean TTFT (ms):                          97.13     
Median TTFT (ms):                        66.12     
P99 TTFT (ms):                           311.50    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          38.85     
Median TPOT (ms):                        39.36     
P99 TPOT (ms):                           47.97     
==================================================
```
