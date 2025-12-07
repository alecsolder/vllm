from vllm import LLM, SamplingParams
llm = LLM(model='facebook/opt-125m', enable_sleep_mode=True)
prompts = ['Hello, my name is']
sampling_params = SamplingParams(temperature=0, max_tokens=10)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
