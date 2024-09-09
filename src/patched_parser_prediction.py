import os
from bs4 import BeautifulSoup
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import torch
from tqdm import tqdm
import csv
import requests
import base64
import math

def fetch_website_content(filepath):
    encodings = ['utf-8', 'latin-1']
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as file:
                content = file.read()
                if content.strip():
                    return content
                else:
                    return None
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Failed to decode file {filepath} with supported encodings.")

# Apply Patches
def apply_parser_modifications(soup):
    for tag in soup.find_all(attrs={'hidden': True}):
        tag.extract()
    for tag in soup.find_all(style=lambda value: value and 'display: none' in value):
        tag.extract()
    
    style_tags = soup.find_all('style')
    styles = ''.join([tag.string for tag in style_tags if tag.string])
    if 'display: none' in styles:
        for tag in soup.find_all():
            if tag.get('style') and 'display: none' in tag.get('style'):
                tag.extract()
    
    for form in soup.find_all('form'):
        action = form.get('action')
        if action and not (action.startswith('#') or requests.utils.urlparse(action).scheme in ['http', 'https']):
            form['action'] = '#invalid-action'
    
    for script in soup.find_all('script'):
        if script.string:
            try:
                script.string.encode('utf-8').decode('utf-8')
            except UnicodeDecodeError:
                try:
                    base64_encoded_str = script.string.split(",")[1]
                    decoded_script = base64.b64decode(base64_encoded_str).decode('utf-8')
                    script.string = decoded_script
                except Exception as e:
                    script.extract()
                    continue

    return soup

def generate_text_representation(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    soup = apply_parser_modifications(soup)
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

def load_model_and_tokenizer(model_dir):
    model = MobileBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = MobileBertTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def predict_with_sliding_window(model, tokenizer, text, window_size, stride):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze()
    total_tokens = tokens.size(0)
    max_phishing_prob = 0.0

    # Sliding window approach
    for i in range(0, total_tokens, stride):
        window = tokens[i:i+window_size]
        if window.size(0) < window_size:
            break  # Ensures the window size matches the required size
        
        inputs = tokenizer.decode(window, skip_special_tokens=True)
        inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        phishing_prob = probabilities[0, 1].item()

        # Update max phishing probability
        if phishing_prob > max_phishing_prob:
            max_phishing_prob = phishing_prob

    # Determine prediction based on maximum phishing probability
    prediction = "phishing" if max_phishing_prob > 0.5 else "benign"
    return prediction, max_phishing_prob

def main():
    model_dir = './model'  # Updated to 'model'
    model, tokenizer = load_model_and_tokenizer(model_dir)

    samples_dir = os.path.expanduser('~/datasets/samples')
    output_csv = 'predictions.csv'

    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "prediction", "confidence_score"])

    files = [f for f in os.listdir(samples_dir) if f.endswith('.txt')]
    progress_bar = tqdm(files, desc="Processing files")

    window_size = 128  # Adjust this based on model capacity
    stride = 64  # Usually half of the window size to allow overlap

    for file in progress_bar:
        file_path = os.path.join(samples_dir, file)
        html_content = fetch_website_content(file_path)
        
        if html_content:
            text_representation = generate_text_representation(html_content)
            prediction, confidence_score = predict_with_sliding_window(model, tokenizer, text_representation, window_size, stride)
            
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([file, prediction, confidence_score])

if __name__ == "__main__":
    main()
