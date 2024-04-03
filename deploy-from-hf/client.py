from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

# Completion API
stream = False
prompt = "The best thing about"
completion = client.completions.create(
    model=model,
    prompt=prompt,
    max_tokens=100,
    temperature=0,
    n=1,
    stream=stream)

if stream:
    for c in completion:
        print(c)
else:
    print(f"\n----- Prompt:\n{prompt}")
    print(f"\n----- Completion:\n{completion.choices[0].text}")