## Deploy models from HF Hub

#### Install SparseML

```bash
python3 -m venv sparseml-env
source sparseml-env/bin/activate
pip install sparseml[transformers]
```

#### Apply SparseGPT

Apply SparseGPT as we did before:

```bash
python3 run_sparsegpt.py
```

## Deploy with nm-vllm

You can either use our Docker image or install via PyPI.

### Docker

Pull down the Docker image:

```bash
docker pull ghcr.io/neuralmagic/nm-vllm-openai:v0.1.0
docker tag ghcr.io/neuralmagic/nm-vllm-openai:v0.1.0 nm-vllm:v0.1.0
```

#### Deploy the model:

```bash
docker run \
    --gpus all \
    --shm-size 2g \
    -p 8000:8000 \
    -v $PWD/very-tiny-llama-pruned:/data/very-tiny-llama-pruned \
    nm-vllm:v0.1.0 --model /data/very-tiny-llama-pruned --sparsity sparse_w16a16
```

### PyPI

Install nm-vllm:

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install nm-vllm[sparse]
```

Launch the server:

```bash
python3 -m vllm.entrypoints.openai.api_server --model ./very-tiny-llama-pruned --sparsity sparse_w16a16
```

Note: we can also run via Python.

```python
from vllm import LLM, SamplingParams
model = LLM("./very-tiny-llama-pruned", sparsity="sparse_w16a16")
output = model.generate("Mario jumped", SamplingParams(max_tokens=10, temperature=0))
print(output[0].outputs[0].text)
```

## Query the model

Install OpenAI client library:
```bash
python3 -m venv client-env
source client-env/bin/activate
pip install openai
```

Query:

```bash
python3 client.py
```