# PhishLang

**Repository for paper**: *PhishLang: A Lightweight, Client-Side Phishing Detection Framework using MobileBERT for Real-Time, Explainable Threat Mitigation* [(arXiv link)](https://arxiv.org/abs/2408.05667)

PhishLang is an open-source, lightweight language model (based on MobileBERT) specifically designed for phishing website detection through contextual analysis of website content.

This repository includes:
- **PhishLang Model**: Implementation of the PhishLang model for detecting phishing websites.
- **Client-Side Browser Extension**: A client-side implementation of PhishLang as a browser extension for Chromium-based browsers, enabling local inference without relying on online blocklists. The extension runs efficiently even on low-end systems, with no noticeable impact on browsing speed.

### Cite our paper:
```
@article{roy2024utilizing, 
title={Utilizing Large Language Models to Optimize the Detection and Explainability of Phishing Websites}, 
author={Roy, Sayak Saha and Nilizadeh, Shirin}, 
journal={arXiv preprint arXiv:2408.05667}, 
year={2024} }
```

## Client-side app: Installation Instructions

### Step 1: Install the Client-Side Service

1. Navigate to the folder 'phishlang_clientside_app' containing `installer.deb`.
2. Enter the following command in the terminal:

   sudo dpkg -i phishlang.deb

   OR

   Double-click on the `.deb` file and install it.

**Note:** It may take some time (2-5 minutes on a 100 Mbps connection) as it downloads dependencies. During this time, the installer may show the status as "Preparing" and might seem stuck. Please do not abort the installation.  
Tested on: Ubuntu 22.04 LTS / 24.04 LTS.

After installation, PhishLang will run in the background. To check its status, use the following command:

   sudo systemctl status phishlang.service

### Step 2: Install the Web Extension

For any Chromium-based browser (Google Chrome, Brave, Microsoft Edge, etc.):

1. Go to `Extensions` -> Turn on `Developer mode`.
2. Drag and drop the `phishlang_extension.crx` file into the Extension window.

Screenshots:

# ![Alt text](/phishlang_clientside_app/screenshots/warning_page.png?raw=true "PhishLang Warning page")
# ![Alt text](/phishlang_clientside_app/screenshots/popup_menu.png?raw=true "PhishLang popup menu")


## Running the Model

1. Install dependencies by running: 
   pip3 install -r requirements.txt

2. Combine the training data into one single file. Navigate to folder training_data and run: cat parsed_samples_part_* > parsed_samples_combined.zip

3. Start the training process: 
   python3 training.py full

**Optional:** 
To divide each sample into 128-token chunks, run: 
   python3 training.py full chunk

*Note: This option is not recommended as it significantly slows down the process with minimal performance improvement for MobileBERT.*

## Running Adversarial Attacks

1. Prepare adversarial samples by running: 
   python3 adversarial_choose_primary_attack/choose_primary_attack.py 
   Wait until the adversarial samples are ready.

2. Patch and predict using adversarial samples by running: 
   python3 patched_parser_prediction.py 
   This should be executed from the root of the directory.





