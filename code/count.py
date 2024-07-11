def count_commas(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
    return text.count(',')

# Usage
file_path = 'text.txt'  # replace with your file path
print(f"The file '{file_path}' contains {count_commas(file_path)} commas.")
