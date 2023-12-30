import os
from tqdm import tqdm
def merge_txt_files(root_folder, output_file):
    txt_file_paths = []

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".txt"):
                txt_file_paths.append(os.path.join(foldername, filename))
    combined_text = ""
    for txt_file_path in tqdm(txt_file_paths):
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            file_text = file.read()
            combined_text += file_text + "\n\n"

    with open(output_file, 'w', encoding='utf-8') as output:
        output.write(combined_text)

if __name__ == "__main__":
    root_folder = "russ_lit" #исходный датасет
    output_file = "all_text.txt"
    merge_txt_files(root_folder, output_file)
