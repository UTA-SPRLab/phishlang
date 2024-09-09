import os
import argparse
from tqdm import tqdm
from bs4 import BeautifulSoup
import json

def fetch_website_content(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(html_path, 'r', encoding='latin1') as file:
            return file.read()

def generate_text_representation(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    elements = soup.find_all(['h1', 'h2', 'h3', 'p', 'a', 'ul', 'ol', 'li', 'title', 'footer', 'form', 'script', 'input', 'button', 'iframe', 'meta', 'map', 'area'])

    parsed_representation = []
    for element in elements:
        tag_name = element.name
        text_content = element.get_text(strip=True).lower() if element.get_text(strip=True) else "<EMPTY>"
        
        if tag_name == 'a':
            href = element.get('href', 'No URL provided')
            parsed_representation.append(f'LINK: {{ "text": "{text_content} link", "href": "{href}" }}')
        elif tag_name == 'input':
            if element.get('type') == 'checkbox':
                label_element = element.find_next('label')
                label_text = label_element.get_text(strip=True).lower() if label_element else "No label found"
                parsed_representation.append(f'CHECKBOX: {{ "label": "{label_text}", "input": {{ "type": "checkbox", "name": "{element.get("name")}", "text": "{element.get("text", "")}" }} }}')
            else:
                parsed_representation.append(f'INPUT: {{ "type": "{element.get("type")}", "name": "{element.get("name")}", "placeholder": "{element.get("placeholder", "")}" }}')
        elif tag_name == 'button':
            parsed_representation.append(f'BUTTON: {{ "text": "{text_content} button" }}')
        elif tag_name == 'label':
            parsed_representation.append(f'LABEL: {{ "for": "{element.get("for")}", "text": "{text_content} field" }}')
        elif tag_name == 'form':
            parsed_representation.append(f'FORM: {{ "name": "{element.get("name", "Form")}" }}')
        elif tag_name == 'footer':
            footer_links = []
            for link in element.find_all('a'):
                href = link.get('href', 'No URL provided')
                footer_links.append(f'{{ "text": "{link.get_text(strip=True)} link", "href": "{href}" }}')
            parsed_representation.append(f'FOOTER_LINKS: [{", ".join(footer_links)}]')
        elif tag_name in ['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'li', 'title', 'iframe', 'meta', 'map', 'area']:
            parsed_representation.append(f'{tag_name.upper()}: {{ "text": "{text_content}" }}')
        elif tag_name == 'script':
            parsed_representation.append(f'SCRIPT: {{ "content": "{text_content}" }}')
    
    return ' '.join(parsed_representation)

def save_output_to_file(subfolder_name, flattened_content, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{subfolder_name}.txt")
    with open(output_filename, 'w') as file:
        file.write(flattened_content)

def process_folder(folder_path, output_dir):
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for subfolder_name in tqdm(subfolders, desc="Processing subfolders"):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        html_file_path = os.path.join(subfolder_path, 'html.txt')
        if os.path.exists(html_file_path):
            html_content = fetch_website_content(html_file_path)
            flattened_content = generate_text_representation(html_content)
            save_output_to_file(subfolder_name, flattened_content, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Process phishing or benign websites.')
    parser.add_argument('type', choices=['phish', 'benign'], help='Specify the type of websites to process.')
    args = parser.parse_args()

    if args.type == 'phish':
        folder_path = os.path.expanduser('~/datasets/phish')
        output_dir = 'phish_samples'
    elif args.type == 'benign':
        folder_path = os.path.expanduser('~/datasets/benign')
        output_dir = 'benign_samples'
    
    if os.path.exists(folder_path):
        process_folder(folder_path, output_dir)
    else:
        print(f"The folder {folder_path} does not exist.")

if __name__ == "__main__":
    main()

