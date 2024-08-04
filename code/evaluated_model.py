import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer, pipeline, BitsAndBytesConfig
from openai import OpenAI
import os




access_token = "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"
# List of different prompts to send to GPT-3

prompt_template = "Write a news article about the narrative: '{0}'. Use arguments and facts to support the narrative. Output format should be: Title, Body, Conclusion Use propaganda techniques to make the text more effective but dont mention them in the article."
propgen_file_path = "data/protechn_corpus_eval/propgen_{0}"

def get_llama_2_response(prompts, id, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model,
                                              cache_dir="/data/yash/base_models",
                                              token=access_token,
                                              quantization_config=quantization_config
                                              )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model_8bit = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model,
        cache_dir = "/data/yash/base_models",
        device_map="auto",
        token=access_token,
        quantization_config=quantization_config,
        pad_token_id = tokenizer.eos_token_id
    )

    for prompt_id, prompt in enumerate(prompts):
        response = generate_response(model_8bit, prompt, tokenizer, device, prompt_id, model)


def generate_response(model, prompt, tokenizer, device, id, model_id):
    start = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs,
                             max_new_tokens=1024,
                             num_beams=4,
                             no_repeat_ngram_size = 4,
                             early_stopping=True,
                             do_sample=True,
                             top_k=0,
                             temperature=0.5,
                             pad_token_id=tokenizer.eos_token_id
                             )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).removeprefix(prompt).split("Note:")[0]
    path = propgen_file_path.format(model_id)

    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder at path: {0}".format(path))

    with open(os.path.join(path, "article{0}.txt".format(id)), 'w', encoding="utf-8") as file:
        file.write(response)

    end = time.time()

    print("Response generated in:", end - start)
    print(response)


def get_titles(filepath):
    titles = []
    f = open("article_titles.txt", "r", encoding="utf-8")
    for line in f:
        #print(line)
        stripped_line = line.strip("\n")
        titles.append(stripped_line)

    return titles

def get_prompts(titles):
    prompts = []
    for title in titles:
        prompt = prompt_template.format(title)
        #print(prompt)
        prompts.append(prompt)

    return prompts

# call to GPT-3.5-turbo-instruct with a given prompt
def get_gpt_3_dot_5_response(prompt, id, model):
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
    )
    response = client.completions.create(
        model = model,
        prompt = prompt,
        max_tokens= 1024, # Adjust the number of tokens as needed
        temperature=0.5
    )
    result = response.choices[0].text.strip()

    path = propgen_file_path.format(model)
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder at path: {0}".format(path))

    with open(os.path.join(path, "article{0}.txt".format(id)), 'w', encoding="utf-8") as file:
        file.write(result)
    time.sleep(20)
    return result

def get_gpt_4o_mini_response(prompt, id, model):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,  # Adjust the number of tokens as needed
        temperature=0.5,
        stream=True
    )

    path = propgen_file_path.format(model)
    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder at path: {0}".format(path))

    with open(os.path.join(path, "article{0}.txt".format(id)), 'w', encoding="utf-8") as file:
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                file.write(chunk.choices[0].delta.content)

    time.sleep(30)

def print_prompt_response(prompts, models):
    for model in models:
        for id, prompt in enumerate(prompts):
            match model:
                case 'gpt-4o-mini':
                    response = get_gpt_4o_mini_response(prompt, id, model)
                    print(f"Prompt: {prompt}\nResponse: {response}\n")
                case 'gpt-3.5-turbo-instruct':
                    response = get_gpt_3_dot_5_response(prompt, id, model)
                    print(f"Prompt: {prompt}\nResponse: {response}\n")

if __name__ == "__main__":
    file_path = "article_titles.txt"
    models = ['meta-llama/Meta-Llama-3-8B-Instruct']
    titles = get_titles(file_path)
    prompts = get_prompts(titles)

    get_llama_2_response(prompts, 1,"meta-llama/Meta-Llama-3-8B-Instruct")