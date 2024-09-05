import time
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer, pipeline, BitsAndBytesConfig
from openai import OpenAI
import os




access_token = "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"
# List of different prompts to send to GPT-3

prop_prompt_template = "Write a news article about the narrative: '{0}'. Use arguments and facts to support the narrative. Output format should be: Title, Body, Conclusion Use propaganda techniques to make the text more effective but dont mention them in the article."
base_prompt_template = "Write a news article about the narrative: '{0}'. Use arguments and facts to support the narrative. Output format should be: Title, Body, Conclusion."
propgen_file_path = "data/protechn_corpus_eval/propgen_{0}-{1}"

def get_llama_3_models_response(prompts, model):
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

def setup_llama_model(prompts, model, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model,
                                              cache_dir="/data/yash/base_models",
                                              token=access_token,
                                              quantization_config=quantization_config
                                              )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model_8bit = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model,
        cache_dir = "/data/yash/base_models",
        device_map="auto",
        token=access_token,
        quantization_config=quantization_config,
        pad_token_id = tokenizer.eos_token_id
    )

    for prompt_id, prompt in enumerate(prompts):
        response = generate_response(model_8bit, prompt, tokenizer, device, prompt_id, model, mode)


def post_process_response(response):
    remarks = ["Note:", "Propaganda techniques used:"]
    for remark in remarks:
        if remark in response:
            response = response.split(remark)[0]
    return response

def generate_response(model, prompt, tokenizer, device, id, model_id, mode):
    start = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs,
                             max_new_tokens=1024,
                             do_sample=True,
                             repetition_penalty = 1.2,
                             temperature=0.5,
                             num_beams = 5,
                             top_k = 50,
                             top_p = 0.95,
                             pad_token_id=tokenizer.eos_token_id
                             )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).removeprefix(prompt)

    processed_response = post_process_response(response)

    path = propgen_file_path.format(model_id, mode)

    print(path)
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder at path: {0}".format(path))

    with open(os.path.join(path, "article{0}.txt".format(id)), 'w', encoding="utf-8") as file:
        file.write(processed_response)

    end = time.time()

    print("Response generated in:", end - start)
    print(processed_response)


def get_titles(filepath):
    titles = []
    f = open(filepath, "r", encoding="utf-8")
    for line in f:
        stripped_line = line.strip("\n")
        titles.append(stripped_line)

    return titles

def get_prompts(titles, template):
    prompts = []
    for title in titles:
        prompt = template.format(title)
        prompts.append(prompt)

    return prompts

# call to GPT-3.5-turbo-instruct with a given prompt
def get_gpt_3_dot_5_response(prompts, model, mode):
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
    )
    for id, prompt in enumerate(prompts):
        response = client.completions.create(
            model = model,
            prompt = prompt,
            max_tokens= 1024, # Adjust the number of tokens as needed
            temperature=0.5
        )
        processed_response = post_process_response(response.choices[0].text.strip())

        path = propgen_file_path.format(model, mode)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
            print("Created folder at path: {0}".format(path))

        with open(os.path.join(path, "article{0}.txt".format(id)), 'w', encoding="utf-8") as file:
            file.write(processed_response)
        time.sleep(20)

def get_gpt_4o_mini_response(prompts, model, mode):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    for id, prompt in enumerate(prompts):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        processed_response = post_process_response(response.choices[0].message.content)

        path = propgen_file_path.format(model, mode)

        if not os.path.exists(path):
            os.makedirs(path)
            print("Created folder at path: {0}".format(path))

        with open(os.path.join(path, "article{0}.txt".format(id)), 'w', encoding="utf-8") as file:
            file.write(processed_response)

        time.sleep(20)

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
                case 'meta-llama/Llama-2-7b-chat-hf':
                    response = setup_llama_model(prompt, model)
                    print(f"Prompt: {prompt}\nResponse: {response}\n")

if __name__ == "__main__":
    file_path = "article_titles.txt"
    test_file_path = "test_articles.txt"
    models = ['gpt-4-turbo', 'gpt-4o-mini', 'gpt-4o' ,'gpt-3.5-turbo']
    models_gpt_3 = ['gpt-3.5-turbo-instruct']
    titles = get_titles(file_path)
    test_titles = get_titles(test_file_path)
    prop_prompts = get_prompts(test_titles, prop_prompt_template)
    base_prompts = get_prompts(test_titles, base_prompt_template)

    #setup_llama_model(prop_prompts, "meta-llama/Llama-2-7b-chat-hf", 'prop-test')
    setup_llama_model(prop_prompts, "meta-llama/Meta-Llama-3.1-8B-Instruct", 'prop-test')
    # for model in [models]:
    #     get_gpt_4o_mini_response(prop_prompts, model, 'prop-test')
    #     get_gpt_4o_mini_response(base_prompts, model, 'base-test')
    #
    # for model in models_gpt_3:
    #     get_gpt_3_dot_5_response(prop_prompts, model, 'prop-test')
    #     get_gpt_3_dot_5_response(base_prompts, model, 'base-test')


    #get_gpt_4o_mini_response(prompts, "gpt-4-turbo")
    #get_gpt_4o_mini_response(prop_prompts, 'gpt-4o-mini', 'prop')
    #get_gpt_4o_mini_response(base_prompts, 'gpt-4o-mini', 'base')


