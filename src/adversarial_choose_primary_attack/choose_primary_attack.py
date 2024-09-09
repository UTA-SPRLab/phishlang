import os
import sys
from bs4 import BeautifulSoup
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import torch
from tqdm import tqdm
import difflib
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from extract_features_html import isInternal, getObjects
from manipulations import InjectIntElem, InjectIntElemFoot, InjectIntLinkElem, InjectExtElem, InjectExtElemFoot, UpdateForm, \
                          ObfuscateExtLinks, UpdateHiddenDivs, UpdateHiddenButtons, UpdateHiddenInputs, UpdateIntAnchors, \
                          InjectFakeCopyright, UpdateTitle, ObfuscateJS, UpdateIFrames, InjectFakeFavicon, InjectHiddenForms, \
                          UpdateInputPasswd, InjectJS, InjectIntElemThreshold

SR_manipulations = [
    UpdateHiddenDivs(), UpdateHiddenButtons(), UpdateHiddenInputs(),
    UpdateTitle(), InjectFakeCopyright(), ObfuscateJS(), InjectHiddenForms(),
    UpdateInputPasswd()
]

MR_manipulations = [
    InjectIntElem(), InjectIntElemFoot(), InjectIntLinkElem(), InjectExtElem(), InjectExtElemFoot(), 
    UpdateForm(), ObfuscateExtLinks(), UpdateIntAnchors(), UpdateIFrames(), InjectIntElemThreshold()
]

source_folder = os.path.expanduser("~/datasets/phish")
destination_folder = os.path.join(os.getcwd(), "adversarial")
error_log_path = os.path.join(destination_folder, "error.csv")

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

error_log = []

# Load MobileBERT
def load_model_and_tokenizer(model_dir):
    model = MobileBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = MobileBertTokenizer.from_pretrained(model_dir)
    return model, tokenizer

model_dir = './model'  # Updated to use MobileBERT and model folder name 'model'
model, tokenizer = load_model_and_tokenizer(model_dir)

def compute_confidence_score(html):
    text_representation = generate_text_representation(html)
    chunk_predictions = sliding_window_predict(model, tokenizer, text_representation, 512, 256)
    confidence_score = max(score for _, score in chunk_predictions)
    return confidence_score

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

def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)

    phishing_prob = probabilities[0, 1].item()  # Assuming class 1 is phishing
    return phishing_prob

# Sliding window prediction
def sliding_window_predict(model, tokenizer, text, window_size, step_size):
    total_length = len(text)
    predictions = []
    for start in range(0, total_length, step_size):
        end = min(start + window_size, total_length)
        chunk = text[start:end]
        confidence_score = predict(model, tokenizer, chunk)
        predictions.append((chunk, confidence_score))
        if end == total_length:
            break
    return predictions

def save_html(content, path):
    with open(path, "w", encoding="utf-8") as file:
        file.write(str(content))

# Generate a diff between two HTMLs to identify change in perturbed sample
def generate_diff(original, manipulated):
    original_lines = original.splitlines()
    manipulated_lines = manipulated.splitlines()
    diff = difflib.unified_diff(original_lines, manipulated_lines, lineterm='', fromfile='original', tofile='manipulated')
    return '\n'.join(diff)

subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
progress_bar = tqdm(subfolders, desc="Processing subfolders")

for subfolder in progress_bar:
    html_path = os.path.join(subfolder, "html.txt")
    if os.path.exists(html_path):
        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as file:
                original_html = file.read()
        except Exception as e:
            print(f"Error reading {html_path}: {e}")
            continue

        soup = BeautifulSoup(original_html, "html.parser")

        folder_name = os.path.basename(subfolder)
        adversarial_subfolder = os.path.join(destination_folder, folder_name)
        os.makedirs(adversarial_subfolder, exist_ok=True)

        save_html(soup, os.path.join(adversarial_subfolder, "original.html"))

        # Initial confidence score
        original_score = compute_confidence_score(str(soup))

        # Initialize the best adversarial example and score
        best_example = soup
        best_score = original_score

        # Apply SR manipulations
        sr_meta_data = []
        for manipulation in SR_manipulations:
            try:
                manipulated_html = manipulation(best_example, "https://malicious.io")
                manipulated_score = compute_confidence_score(str(manipulated_html))
                if manipulated_score < best_score:
                    best_example = manipulated_html
                    best_score = manipulated_score
                diff = generate_diff(str(soup), str(manipulated_html))
                sr_meta_data.append({"Manipulation": str(manipulation), "Change": diff, "Score": manipulated_score})
            except Exception as e:
                print(f"Error applying {str(manipulation)} on {folder_name}: {e}")
                error_log.append({"website": folder_name, "type": str(manipulation), "error": str(e)})

        # Save SR meta data to CSV
        sr_meta_df = pd.DataFrame(sr_meta_data)
        sr_meta_df.to_csv(os.path.join(adversarial_subfolder, "sr_meta.csv"), index=False)

        Q = 35  # Query budget
        R = (Q - len(SR_manipulations)) // len(MR_manipulations)

        # Apply MR manipulations
        mr_meta_data = []
        for _ in range(R):
            for manipulation in MR_manipulations:
                try:
                    manipulated_html = manipulation(best_example, "https://malicious.io")
                    manipulated_score = compute_confidence_score(str(manipulated_html))
                    if manipulated_score < best_score:
                        best_example = manipulated_html
                        best_score = manipulated_score
                    diff = generate_diff(str(soup), str(manipulated_html))
                    mr_meta_data.append({"Manipulation": str(manipulation), "Change": diff, "Score": manipulated_score})
                except Exception as e:
                    print(f"Error applying {str(manipulation)} on {folder_name}: {e}")
                    error_log.append({"website": folder_name, "type": str(manipulation), "error": str(e)})

        # Save MR meta data
        mr_meta_df = pd.DataFrame(mr_meta_data)
        mr_meta_df.to_csv(os.path.join(adversarial_subfolder, "mr_meta.csv"), index=False)

        # Identify primary attack(s)
        all_meta_data = sr_meta_data + mr_meta_data
        if all_meta_data:
            meta_df = pd.DataFrame(all_meta_data)
            mean_advantage = meta_df["Score"].mean()
            std_advantage = meta_df["Score"].std()
            primary_attacks = meta_df[meta_df["Score"] <= mean_advantage + std_advantage]
            primary_attacks.to_csv(os.path.join(adversarial_subfolder, "primary_attacks.csv"), index=False)

        # Save the best adversarial HTML
        save_html(best_example, os.path.join(adversarial_subfolder, "best_adversarial.html"))

if error_log:
    error_df = pd.DataFrame(error_log)
    error_df.to_csv(error_log_path, index=False)
