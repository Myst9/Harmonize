# Harmonize
### A Tool to Detect Cyberbullying and Check the Toxicity of Comments on Various Websites
## About
Harmonize is a Google Chrome Extension built to make Software Engineering online spaces more peaceful and harmonious than they currently are. It focuses on giving young and budding software engineers a positive environment so that they can grow and build successful careers. It primarily targets Software Developer websites like GitHub, detects Cyberbullying on these websites and protects you from it. It also assists you in turning your negative/toxic comments into something more constructive/productive before you post comments on the aforementioned sites.
<br />

<p align="center">
<img src="https://github.com/Myst9/Harmonize/blob/main/Images/logo.jpg" />
</p>

## Technical Details

•	BERT Model fine-tuned on ToxiCR dataset  
•	Gemini API for comment suggestion

UI -
1. Vanilla JS
2. Chart JS

## Pipeline

The pipeline/architecture/flow diagram of our toxicity detector model is:

<p align="center">
<img src="https://github.com/Myst9/Harmonize/blob/main/Images/Harmonize_Pipeline.jpg" />
</p>

## Major Features

### Toxicity Chart for GitHub

![image](https://github.com/Myst9/Harmonize/blob/main/Images/image1.png)
![image](https://github.com/Myst9/Harmonize/blob/main/Images/image2.png)

### Autosuggestor feature for GitHub

![image](https://github.com/Myst9/Harmonize/blob/main/Images/image3.png)

### Toxicity Chart for Gmail

![image](https://github.com/Myst9/Harmonize/blob/main/Images/image4.jpg)

### Autosuggestor feature for Gmail

![image](https://github.com/Myst9/Harmonize/blob/main/Images/image5.jpg)

## Instructions
1. Clone the repository using the following commands
```
$ git lfs install
$ git clone git@github.com:Myst9/Harmonize.git
$ cd Harmonize
$ git lfs fetch
$ git lfs checkout
```
2. Enter the directory Harmonize and type 
```
$ npm install
$ pip install requirements.txt
```
3. Navigate to the API folder
4. Edit the API key inside the test.py file with your Gemini API key.
5. Run the following command
```
$ python test.py
```
6. Go to chrome browser and type 
```
chrome://extensions/
```
7. Turn on Developer mode  
8. Click on Load Unpacked option and browse to the folder Client and select it.  
9. Enable/Reload the extension  
10. Navigate to GitHub or Gmail  
