# Spam Message Detector using Machine Learning

Intermediate Python project that classifies text messages as **spam** or **ham (not spam)** using TF-IDF, Logistic Regression, and Streamlit.

## Features
- Interactive message prediction
- TF-IDF text features with unigrams and bigrams
- Logistic Regression classifier
- Train/test split and evaluation report
- Confusion matrix and included sample dataset

## Project structure
```text
spam-message-detector/
├── app.py
├── data/
│   └── messages.csv
├── outputs/
│   └── sample_predictions.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Run on Windows 11 with VS Code
1. Install Python 3.9+ and VS Code.
2. Extract the ZIP and open the project folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment: `py -m venv .venv`
5. Activate in PowerShell: `.venv\\Scripts\\Activate.ps1`
   - If activation is blocked, use Command Prompt: `.venv\\Scripts\\activate.bat`
6. Install dependencies: `py -m pip install -r requirements.txt`
7. Run: `py -m streamlit run app.py`
8. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## How it works
1. Pandas loads the labelled CSV file.
2. A stratified train/test split separates training and evaluation records.
3. TF-IDF converts text into numeric features.
4. Logistic Regression learns to classify messages.
5. Streamlit displays the prediction and held-out evaluation metrics.

## Dataset and limitations
The included dataset contains 60 hand-written educational examples (30 ham and 30 spam). It is a small demonstration dataset, not a representative real-world corpus. Metrics can be unstable, and a prediction does not prove a message or link is safe.

## Example messages
| Message | Expected label |
|---|---|
| Please send me the notes from today's lecture. | ham |
| I will reach the bus stop in ten minutes. | ham |
| Congratulations! You won a free prize. Claim now! | spam |
| Get a free gift card today. Click here. | spam |

See `outputs/sample_predictions.csv`. These expected labels are examples, not guaranteed model predictions.

## Upload to GitHub using terminal
Create an empty GitHub repository named `spam-message-detector`, then run these commands from the project folder:
```bash
git init
git add .
git commit -m "Add Spam Message Detector ML project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/spam-message-detector.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. Do not upload `.venv`; it is excluded by `.gitignore`.

## Future improvements
- Use a larger public SMS spam dataset.
- Compare Naive Bayes, Logistic Regression, and SVM.
- Add cross-validation and hyperparameter tuning.
- Save the trained pipeline using Joblib.

## Author
Add your name, course, college, and GitHub profile.
