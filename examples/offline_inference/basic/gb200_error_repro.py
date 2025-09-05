# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
# Sample prompts.
prompts = [
    TokensPrompt(prompt_token_ids=[])
] * 25
# Create a sampling params object.
sampling_params = SamplingParams(temperature=1)


def main():
    # Create an LLM.
    llm = LLM(model="/home/alecs/local/checkpoints/gpt-oss-120b")
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        generated_text = output.outputs[0].text
        header = generated_text.split("<|message|>")[0]
        if header.startswith("assistant<|channel|>analysis to=container.exec"):
            print("Header valid, matches assistant<|channel|>analysis to=container.exec")
        else:
            print(f"Header invalid, was: {header}")
        print("-" * 60)


if __name__ == "__main__":
    main()
