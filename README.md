# Harmonize
### A Tool to Detect Cyberbullying and Check the Toxicity of Comments on Various Websites
## About
Harmonize is a Google Chrome Extension which primarily targets software developer websites such as Github. It detects cyberbullying on them, and provides toxicity reports on comments, if a toxic comment is already posted in website, it blurs the comment. Along with this, Harmonize also assists the user in avoiding the use of toxic language by suggesting gentler alternatives as they type.


## Technical Details

•	BERT Model fine-tuned on ToxiCR dataset  
•	Gemini API for comment suggestion

UI -
1. Vanilla JS
2. Chart JS

## Major Features

### Toxicity Chart for GitHub

![image](https://github.com/Myst9/Harmonize/blob/main/Images/image1.png)
![image](https://github.com/Myst9/Harmonize/blob/main/Images/image2.png)

### Autosuggestor feature for GitHub

![image](https://github.com/Myst9/Harmonize/blob/main/Images/image3.png)


## Instructions
1. Clone the repository
2. Enter the directory Harmonize and type 
```
$ npm install
$ pip install requirements.txt
```
3. Navigate to the API folder.
4.	Edit the API key inside the test.py file with your Gemini API key.
5.	Run the following command
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
