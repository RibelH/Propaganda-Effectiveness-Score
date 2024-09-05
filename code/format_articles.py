import os
import re


def reformat_files(filepath):
    with open(filepath, 'r') as file:
        content = file.read()

    # Reformat content
    sentences = re.split(r'(?<=[.!?])\s+', content)
    formatted_content = "\n".join(sentence.strip() for sentence in sentences)

    # Write formatted content back into file
    with open(filepath, 'w') as file:
        file.write(formatted_content)


def reformat_files_in_directory(directory_path):

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)

        if os.path.isfile(filepath):
            reformat_files(filepath)

if __name__ == "__main__":
    # Specify the directory containing the files
    directory_path = './data/protechn_corpus_eval/propgen_meta-llama/Meta-Llama-3.1-8B-Instruct-prop-test'

    # Reformat all files in the specified directory
    reformat_files_in_directory(directory_path)