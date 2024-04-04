# Deploy models from HF Hub

`nm-vllm` can run models directly from the Hugging Face hub with great performance.

In this example we will deploy Mistral.

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
    nm-vllm:v0.1.0 --model mistralai/Mistral-7B-v0.1 --max-model-len 4096
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
python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-v0.1 --max-model-len 4096
```

Note: we can also run via Python.

```python
from vllm import LLM, SamplingParams
model = LLM("mistralai/Mistral-7B-v0.1", sparsity="sparse_w16a16")
output = model.generate("Hello my name is", SamplingParams(max_tokens=50, temperature=0))
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