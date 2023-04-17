# LLaMA.py

> **NOTICE: Deprecation** I originally wrote this script as a makeshift solution before a proper binding came out, and since there are projects like [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) providing working bindings to the latest llama.cpp (which updates faster than I can keep up), I'm no longer planning to maintain this repository and would like to kindly direct interest people to other solutions.

A quick and dirty script to call [llama.cpp](https://github.com/ggerganov/llama.cpp) in Python. Supports streaming and interactive mode.

## Usage

This Python script requires the compiled `main` binary from LLaMA.cpp. You'll need to compile [llama.cpp](https://github.com/ggerganov/llama.cpp) for your own machine, as well as grab a copy of the model weights and quantize them according to instructions provided in llama.cpp.

By default, the script assumes that you have your model weights as `./models/7B/ggml-model-q4_0.bin`, and the llama.cpp binary as `./llama.cpp/main`. However, you can point the script to your own paths.

### Quiet mode
If you just want the end result, without the streaming part:
```python
from llama import llama

output = llama('LLaMA is a large language model that', streaming=False):
print(output)
```

### Streaming mode
Simplest example:
```python
from llama import llama

for token in llama('LLaMA is a large language model that'):
    print(token, end='', flush=True)
```

If you don't want to see the prompt, just the completion:
```python
from llama import llama

for token in llama('LLaMA is a large language model that', skip_prompt=True):
    print(token, end='', flush=True)
```

Additionally, you can choose to show a small tail of the prompt by specifying the character count:
```python
from llama import llama

for token in llama(
    'LLaMA is a large language model that can:\n1.', 
    skip_prompt=True,
    trim_prompt=2, # the '1.' part of the prompt will be shown
):
    print(token, end='', flush=True)
```

### Interactive mode
```python
from llama import llama

for token in llama(
    'Below is a conversation between a user and LLaMA:\nUser: Hello!\nLLaMA: Hi! I am LLaMA, a large language model.\nUser: ',
    interactive=True,
    reverse_prompt="User: "
):
    print(token, end='', flush=True)
```

### Model parameters
Here's the pull range of parameters that you can tweak. As you can see, you can change the executable and model path by supplying the `executable` and `model` parameters.
```python
def llama_stream(
    prompt='',
    skip_prompt=True,
    trim_prompt=0,
    executable='./llama.cpp/main',
    model='./models/7B/ggml-model-q4_0.bin',
    threads=4,
    temperature=0.7,
    top_k=40,
    top_p=0.5,
    repeat_last_n=256,
    repeat_penalty=1.17647,
    n=4096,
    interactive=False,
    reverse_prompt="User:"
)
```
