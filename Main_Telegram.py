from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import tkinter as tk      


# ==============================================================
#  BASIC PYTHON & SYSTEM UTILITIES
# ==============================================================
import os
import json
import base64
import pickle
import warnings
import base64
import requests
warnings.filterwarnings('ignore')

# ==============================================================
#  NUMERICAL & DATA PROCESSING LIBRARIES
# ==============================================================
import numpy as np
import pandas as pd

# ==============================================================
#  IMAGE PROCESSING & VISUALIZATION
# ==============================================================
import cv2                           # OpenCV for image operations
import matplotlib.pyplot as plt       # Plotting / visualization
from PIL import Image, ImageDraw, ImageFont  # Image drawing utilities
from PIL import Image, ImageTk, ImageEnhance
# ==============================================================
#  MACHINE LEARNING (SKLEARN)
# ==============================================================
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
)


from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.utils import resample             
import joblib   # Saving & loading models
import seaborn as sns
import re

# Classifiers
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from figs import FIGS
from sklearn.linear_model import PassiveAggressiveClassifier


# ==============================================================
#  DEiT TRANSFORMER (TIMM + TORCH)
# ==============================================================
import torch
import timm
from torchvision import transforms


# ==============================================================
#  TELEGRAM BOT (python-telegram-bot library)
# ==============================================================
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    MessageHandler, ContextTypes, filters
)



global filename
global X, Y
global model
global categories,model_folder


model_folder = "model"
os.makedirs(model_folder, exist_ok=True)   # create model folder if not exist

feature_file = os.path.join(model_folder, "final_deit_features.npz")

accuracy = []
precision = []
recall = []
fscore = []

def Intialize_DeiT():
    global model, preprocess, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(
        "deit_small_patch16_224",
        pretrained=True,
        num_classes=0
    )
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])



def uploadDataset():
    global filename,categories,categories,path
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")
    
def DeiT_Feature_Extraction():
    global filename, categories,X, Y
    global model,preprocess,device

    Intialize_DeiT()
    text.delete('1.0', END)
    text.insert(END, "Starting DeiT (Data-efficient Image Transformer) Feature Extraction...\n")

    # --------------------------------------------------
    # LOAD FEATURES IF AVAILABLE
    # --------------------------------------------------
    if os.path.exists(feature_file):
        text.insert(END, "Saved feature file found. Loading features from disk...\n")
        data = np.load(feature_file, allow_pickle=True)
        X = data["features"]
        Y = data["labels"]
        text.insert(END, "Loaded features successfully.\n")
        return X, Y

    text.insert(END, "No saved file found. Extracting DeiT features...\n")

    # --------------------------------------------------
    # LOAD DeiT MODEL (from TIMM)
    # --------------------------------------------------

    X_features = []
    Y_labels = []

    for class_name in categories:
        class_folder = os.path.join(filename, class_name)

        text.insert(END, f"Processing Class: {class_name}\n")

        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            print(img_path)

            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            # Load image
            img = Image.open(img_path).convert("RGB")
            img = preprocess(img)
            img = img.unsqueeze(0).to(device)

            # Extract DeiT features
            with torch.no_grad():
                feat = model(img)

            feat_np = feat.cpu().numpy().flatten()

            X_features.append(feat_np)
            Y_labels.append(class_name)

    X_features = np.array(X_features)
    Y_labels = np.array(Y_labels)

    text.insert(END, "Feature extraction complete.\n")
    text.insert(END, f"Total samples processed: {len(Y_labels)}\n")

    np.savez(feature_file, features=X_features, labels=Y_labels)
    text.insert(END, f"Saved features to: {feature_file}\n")

    return X_features, Y_labels
    
def Train_Test_split():
    global X, Y, x_train, x_test, y_train, y_test

    text.delete('1.0', END)
    text.insert(END, "Performing Train-Test split (80/20)...\n")


    
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X = X[indices]
        Y = Y[indices]
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X = X[indices]
        Y = Y[indices]
    

    x_train, x_test, y_train, y_test = train_test_split(
        X, 
        Y,
        test_size=0.20,
        random_state=42,
        stratify = Y# maintain class balance
    )

    # Print results in GUI
    text.insert(END, f"Training dataset size: {x_train.shape}\n")
    text.insert(END, f"Testing dataset size : {x_test.shape}\n")

    text.insert(END, "Train-Test splitting complete.\n")

    return x_train, x_test, y_train, y_test

def Calculate_Metrics( algorithm, predict, y_test,y_score):
    global categories
    if not os.path.exists('results'):
        os.makedirs('results')
        
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure() 
    ax = sns.heatmap(
        conf_matrix,
        xticklabels=categories,
        yticklabels=categories,
        annot=True,
        cmap="viridis",
        fmt="g",
        cbar=False       )

    ax.set_ylim([0, len(categories)])  # Keeps all rows visible
    plt.title(algorithm + " Confusion Matrix", fontsize=14, pad=15)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)

    plt.tight_layout()  # Adjusts subplots to fit within figure area

    # ✅ Save figure without cropping anything
    plt.savefig(
        f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png",
        dpi=300,
        bbox_inches='tight'  # Ensures full content is saved
    )
    plt.show()

    if y_score is not None:

        # Convert output labels to numeric if needed
        class_indices = {c: i for i, c in enumerate(categories)}
        y_test_num = np.array([class_indices[y] for y in y_test])

        # One-vs-Rest binarization
        y_test_bin = label_binarize(y_test_num, classes=range(len(categories)))

        # Number of classes
        n_classes = y_test_bin.shape[1]

        fpr = {}
        tpr = {}
        roc_auc = {}

        # ---------------------------------------------
        # COMPUTE ROC FOR EACH CLASS
        # ---------------------------------------------
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # ---------------------------------------------
        # MICRO-AVERAGE ROC (global)
        # ---------------------------------------------
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # ---------------------------------------------
        # PLOT ROC CURVES
        # ---------------------------------------------
        plt.figure(figsize=(10, 8))

        # Plot each class ROC
        for i in range(n_classes):
            plt.plot(
                fpr[i], tpr[i],
                label=f'{categories[i]} (AUC = {roc_auc[i]:.2f})',
                linewidth=2
            )

        # Plot micro-average
        plt.plot(
            fpr["micro"], tpr["micro"],
            linestyle='--', linewidth=3,
            label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})'
        )

        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.title(f"{algorithm} ROC Curves (One-vs-Rest)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
        plt.show()

        text.insert(END, "\nAUC Scores (per class):\n")
        for i in range(n_classes):
            text.insert(END, f"{categories[i]} : {roc_auc[i]:.4f}\n")

        text.insert(END, f"\nMicro-average AUC: {roc_auc['micro']:.4f}\n\n")

def Existing_Ridge():
    global x_train, x_test, y_train, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "Ridge_model.pkl")

    if os.path.exists(model_filename):
        text.insert(END, "Loading saved Ridge Classifier model...\n")
        mlmodel = joblib.load(model_filename)
    else:
        text.insert(END, "Training new Ridge Classifier...\n")
        mlmodel = RidgeClassifier()

        # Train model
        mlmodel.fit(x_train, y_train)

        # Save model
        joblib.dump(mlmodel, model_filename, compress=5)
        text.insert(END, f"Model saved at: {model_filename}\n")

    y_pred = mlmodel.predict(x_test)
    scores = mlmodel.decision_function(x_test)
    Calculate_Metrics("Existing Ridge Classifier", y_pred, y_test,scores)

def Existing_PAC():
    global x_train, x_test, y_train, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "PAC_model.pkl")

    if os.path.exists(model_filename):
        text.insert(END, "Loading saved Passive-Aggressive Classifier model...\n")
        mlmodel = joblib.load(model_filename)
    else:
        text.insert(END, "Training new Passive-Aggressive Classifier...\n")
        mlmodel = PassiveAggressiveClassifier()

        # Train model
        mlmodel.fit(x_train, y_train)

        # Save model
        joblib.dump(mlmodel, model_filename, compress=5)
        text.insert(END, f"Model saved at: {model_filename}\n")

    y_pred = mlmodel.predict(x_test)
    scores = mlmodel.decision_function(x_test)
    Calculate_Metrics("Existing PAC", y_pred, y_test,scores)

def Existing_NCC():
    global x_train, x_test, y_train, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "NCC_model.pkl")

    if os.path.exists(model_filename):
        text.insert(END, "Loading saved Nearest Centroid Classifier model...\n")
        mlmodel = joblib.load(model_filename)
    else:
        text.insert(END, "Training new Nearest Centroid Classifier...\n")
        nc = NearestCentroid()

        # Use probability calibration
        mlmodel = CalibratedClassifierCV(nc, cv=3)

        mlmodel.fit(x_train, y_train)

        joblib.dump(mlmodel, model_filename, compress=5)
        text.insert(END, f"Model saved at: {model_filename}\n")

    y_pred = mlmodel.predict(x_test)
    y_score = mlmodel.predict_proba(x_test)

    Calculate_Metrics("Existing NCC", y_pred, y_test,y_score)

def Proposed_FIGS():
    global x_train, x_test, y_train, y_test, model_folder
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "FIGS_model.pkl")

    if os.path.exists(model_filename):
        text.insert(END, "Loading saved FIGS model...\n")
        mlmodel = joblib.load(model_filename)
    else:
        text.insert(END, "Training new FIGS...\n")
        mlmodel = FIGS()

        # Train model
        mlmodel.fit(x_train, y_train)
        # Save model
        joblib.dump(mlmodel, model_filename, compress=5)
        text.insert(END, f"Model saved at: {model_filename}\n")

    # Predict
    y_pred = mlmodel.predict(x_test)
    probs = mlmodel.predict_proba(x_test) 

    Calculate_Metrics("Proposed FIGS", y_pred, y_test,probs)


def predict_original():
    global model_folder, categories,preprocess,model

    text.delete('1.0', END)
    text.insert(END, "Starting Prediction on Test Image...\n")

    test_image_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )




    img = Image.open(test_image_path).convert("RGB")
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    text.insert(END, "Extracting features using DeiT...\n")
    with torch.no_grad():
        feature = model(img_tensor)

    feature_np = feature.cpu().numpy().flatten().reshape(1, -1)
    
    model_filename = os.path.join(model_folder, "FIGS_model.pkl")

    final_model = joblib.load(model_filename)
    prediction = final_model.predict(feature_np)[0]
    text.insert(END, f"Prediction Completed: {prediction}\n")

    # --------------------------------------------------
    # DISPLAY RESULT ON IMAGE
    # --------------------------------------------------
    img_cv = cv2.imread(test_image_path)
    img_display = cv2.resize(img_cv, (500, 500))

    cv2.putText(img_display, f'Classified as: {prediction}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Prediction", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    text.insert(END, "Prediction displayed successfully.\n")

def predict_transformer(test_image_path):
    global model_folder, categories,preprocess,model

    img = Image.open(test_image_path).convert("RGB")
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    text.insert(END, "Extracting features using DeiT...\n")
    with torch.no_grad():
        feature = model(img_tensor)

    feature_np = feature.cpu().numpy().flatten().reshape(1, -1)
    
    model_filename = os.path.join(model_folder, "FIGS_model.pkl")

    final_model = joblib.load(model_filename)
    prediction = final_model.predict(feature_np)[0]
    text.insert(END, f"Prediction Completed: {prediction}\n")
    return prediction


def predict():
    global model_folder, categories, transform, model
    Intialize_DeiT()
    

    text.delete('1.0', END)
    text.insert(END, "Starting Prediction on Test Image...\n")

    # --------------------------------------------------
    # STEP 1: SELECT TEST IMAGE
    # --------------------------------------------------
    test_image_path = filedialog.askopenfilename( initialdir="TestImages", 
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    if not test_image_path:
        text.insert(END, "No image selected.\n")
        return

    # --------------------------------------------------
    # STEP 2: RUN XAI X-RAY DETECTION
    # --------------------------------------------------
    result = analyze_xray_or_image(test_image_path)

    # Convert JSON string to dictionary
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            text.insert(END, "Error: XAI returned invalid JSON.\n")
            return

    print("XAI Result:", result)

    # --------------------------------------------------
    # CASE C: XAI FAILED → SHOW ONLY ORIGINAL + TRANSFORMER RESULT
    # --------------------------------------------------
    if "error" in result:
        text.insert(END, "⚠ XAI Failed: Showing Only Transformer Prediction.\n")

        prediction = predict_transformer(test_image_path)
        text.insert(END, f"Classifier Prediction: {prediction}\n")

        plt.figure(figsize=(14, 6))

        # 121 → Original Image
        plt.subplot(121)
        orig = cv2.imread(test_image_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (500, 500))
        plt.imshow(orig)
        plt.title("Original Image")
        plt.axis("off")

        # 122 → Transformer Prediction
        plt.subplot(122)
        img_cls = cv2.imread(test_image_path)
        img_cls = cv2.cvtColor(img_cls, cv2.COLOR_BGR2RGB)
        img_cls = cv2.resize(img_cls, (500, 500))
        cv2.putText(img_cls, f"Predicted: {prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        plt.imshow(img_cls)
        plt.title("Transformer Prediction")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        return

    # --------------------------------------------------
    # CASE A: X-RAY → Use Your Classifier
    # --------------------------------------------------
    if result.get("is_xray") == True:
        text.insert(END, "Bone X-Ray detected. Running model classifier...\n")

        # CLASSIFIER PREDICTION
        prediction = predict_transformer(test_image_path)
        text.insert(END, f"Classifier Prediction: {prediction}\n")

        # ---------- CREATE X-RAY ANALYSIS WHITE PANEL ----------
        width, height = 800, 600
        result_img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(result_img)

        try:
            font_big = ImageFont.truetype("arial.ttf", 28)
        except:
            font_big = ImageFont.load_default()

        y = 30
        draw.text((30, y), "X-Ray Analysis Results", fill="black", font=font_big)
        y += 50

        for key, value in result.items():
            draw.text((30, y), f"{key}: {value}", fill="black", font=font_big)
            y += 40

        result_np = np.array(result_img)

        # ---------- DISPLAY 3 SUBPLOTS ----------
        plt.figure(figsize=(18, 6))

        # 131 → Original X-Ray
        plt.subplot(131)
        orig = cv2.imread(test_image_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (500, 500))
        plt.imshow(orig)
        plt.title("Original X-Ray")
        plt.axis("off")

        # 132 → XAI X-Ray Result Panel
        plt.subplot(132)
        plt.imshow(result_np)
        plt.title("XAI X-Ray Result")
        plt.axis("off")

        # 133 → Classifier Result
        plt.subplot(133)
        img_cls = cv2.imread(test_image_path)
        img_cls = cv2.cvtColor(img_cls, cv2.COLOR_BGR2RGB)
        img_cls = cv2.resize(img_cls, (500, 500))
        cv2.putText(img_cls, f"Classified: {prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        plt.imshow(img_cls)
        plt.title("Model Classification Result")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        return

    # --------------------------------------------------
    # CASE B: NOT AN X-RAY → NORMAL XAI OUTPUT
    # --------------------------------------------------
    else:
        text.insert(END, f"Not an X-Ray. Detected Image Type = {result.get('image_type')}\n")

        width, height = 800, 600
        result_img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(result_img)

        try:
            font_big = ImageFont.truetype("arial.ttf", 28)
        except:
            font_big = ImageFont.load_default()

        y = 30
        draw.text((30, y), "Image Analysis Result", fill="black", font=font_big)
        y += 50

        for key, value in result.items():
            draw.text((30, y), f"{key}: {value}", fill="black", font=font_big)
            y += 40

        result_np = np.array(result_img)

        # ---------- DISPLAY 2 SUBPLOTS ----------
        plt.figure(figsize=(14, 6))

        # 121 → Original Image
        plt.subplot(121)
        orig = cv2.imread(test_image_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (500, 500))
        plt.imshow(orig)
        plt.title("Original Image")
        plt.axis("off")

        # 122 → XAI Result
        plt.subplot(122)
        plt.imshow(result_np)
        plt.title("Detected Image Type")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        return

 
    
def analyze_xray_or_image(image_path):
    api_key = os.getenv("GEMINI_API_KEY")
    try:
        # Check if the image file exists
        if not os.path.exists(image_path):
            return json.dumps({"error": "File not found"})
        
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Validate API key
        if not api_key:
            return json.dumps({"error": "API key is missing"})

        # XAI API endpoint
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={api_key}"
        )

        # ------------------------------
        # MAIN REQUEST TO XAI
        # ------------------------------
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_data}},
                        {
                            "text": (
                                "Analyze the uploaded image.\n\n"
                                "**TASK 1 — Image Type Detection**:\n"
                                "Identify whether the given image is an X-ray image of a human bone.\n"
                                "If it is NOT a bone X-ray, return what the image contains (example: food, person, animal, chest x-ray, CT scan, etc.).\n\n"
                                "**TASK 2 — If Bone X-Ray**:\n"
                                "If the image *is* a bone X-ray, determine:\n"
                                " - Whether a fracture exists\n"
                                " - Type of fracture (example: transverse, oblique, comminuted, spiral, greenstick, hairline)\n"
                                " - Severity level (mild, moderate, severe)\n"
                                " - Which bone is affected (example: femur, tibia, humerus, radius, ulna, finger, etc.)\n\n"
                                "Return output in STRICT JSON with the following structure:\n"
                                "{\n"
                                "  \"is_xray\": true/false,\n"
                                "  \"image_type\": \"Bone X-Ray\" or \"Food\" or \"Animal\" etc,\n"
                                "  \"bone\": \"Femur\" or null,\n"
                                "  \"fracture_present\": true/false,\n"
                                "  \"fracture_type\": \"Transverse\" or null,\n"
                                "  \"severity\": \"Mild/Moderate/Severe\" or null\n"
                                "}\n\n"
                                "Return ONLY JSON. No explanation text."
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 5000
            }
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        response_json = response.json()

        # Validate API response structure
        if "candidates" not in response_json or not response_json["candidates"]:
            return json.dumps({"error": "No candidates in API response"})

        candidate = response_json["candidates"][0]

        # Extract text response
        if (
            "content" in candidate
            and "parts" in candidate["content"]
            and len(candidate["content"]["parts"]) > 0
        ):
            text_response = candidate["content"]["parts"][0].get("text", "")
        else:
            return json.dumps({"error": "Unexpected response format"})

        # Extract JSON pattern from response
        json_matches = re.findall(r"\{.*\}", text_response, re.DOTALL)
        if not json_matches:
            return json.dumps({"error": "No JSON found in API response"})

        try:
            result_json = json.loads(json_matches[0])
        except json.JSONDecodeError:
            return json.dumps({"error": "JSON decode failed"})

        return json.dumps(result_json, indent=4)

    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"API request failed: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})
        


def close():
    main.destroy()
    
# ---------------------------------------------------------------
#  When user sends an image
# ---------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome! Send me an X-Ray image and I will analyze it.\n\n"
        "I can detect:\n"
        "✔ Whether it is X-Ray or normal image\n"
        "✔ Fracture detection\n"
        "✔ Bone type\n"
        "✔ Severity\n"
        "✔ Your AI Model Prediction"
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):

    # Save incoming image
    photo = update.message.photo[-1]
    file = await photo.get_file()
    file_path = "input_image.jpg"
    await file.download_to_drive(file_path)

    await update.message.reply_text("🔍 Analyzing image... Please wait...")

    # ----------------------------------------------------
    #  RUN YOUR XAI + CLASSIFIER PIPELINE
    # ----------------------------------------------------
    result = analyze_xray_or_image(file_path)

    # Convert JSON response
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            await update.message.reply_text("❌ XAI returned invalid JSON.")
            return

    # ----------------------------------------------------
    # CASE C → XAI FAILED → SEND ONLY TRANSFORMER PREDICTION
    # ----------------------------------------------------
    if "error" in result:
        await update.message.reply_text(
            f"⚠ XAI Failed:\n{result['error']}\n\n"
            "🤖 Sending only AI classifier prediction..."
        )

        # Run your classifier
        prediction = predict_transformer(file_path)

        # Send the result
        await update.message.reply_text(
            f"🤖 **Model Prediction:** {prediction}",
            parse_mode="Markdown"
        )

        # Also send the original image
        await update.message.reply_photo(photo=open(file_path, "rb"))
        return

    # ----------------------------------------------------
    # CASE A → IF IMAGE IS X-RAY → RUN YOUR MODEL
    # ----------------------------------------------------
    if result.get("is_xray") == True:

        prediction = predict_transformer(file_path)

        # Create result panel
        width, height = 800, 600
        panel = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(panel)

        y = 40
        for key, value in result.items():
            draw.text((40, y), f"{key}: {value}", fill="black")
            y += 40

        panel_path = "analysis.jpg"
        panel.save(panel_path)

        # A) Send XAI result
        await update.message.reply_text(
            f"🦴 **X-Ray Detected**\n"
            f"Bone: {result.get('bone')}\n"
            f"Fracture: {result.get('fracture_present')}\n"
            f"Type: {result.get('fracture_type')}\n"
            f"Severity: {result.get('severity')}\n\n"
            f"🤖 Classifier Prediction: {prediction}",
            parse_mode="Markdown"
        )

        # B) Send analysis image
        await update.message.reply_photo(photo=open(panel_path, "rb"))
        return

    # ----------------------------------------------------
    # CASE B → NOT X-RAY
    # ----------------------------------------------------
    else:
        await update.message.reply_text(
            f"📸 Not an X-Ray.\n"
            f"Detected Image Type: {result.get('image_type')}"
        )

        width, height = 800, 600
        panel = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(panel)

        y = 40
        for key, value in result.items():
            draw.text((40, y), f"{key}: {value}", fill="black")
            y += 40

        panel_path = "normal.jpg"
        panel.save(panel_path)
        await update.message.reply_photo(photo=open(panel_path, "rb"))



def close():
    main.destroy()
    
def Telegram_Bot():
    global model_folder, categories, transform, model
    text.insert(END, "🤖 Telegram Bot is running......\n")
    Intialize_DeiT()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set the TELEGRAM_BOT_TOKEN environment variable before starting the bot.")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    print("🤖 Telegram Bot is running...")

    app.run_polling()




main = Tk('Telegram Bot')
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
window_width = int(screen_width)
window_height = int(screen_height )
main.geometry(f"{window_width}x{window_height}")

bg_image = Image.open("background.png")  # replace with your path
bg_image = bg_image.resize((screen_width, screen_height))
bg_photo = ImageTk.PhotoImage(bg_image)

background_label = tk.Label(main, image=bg_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


font = ('times', 15, 'bold')
title = Label(main, text='Telegram Bot Integrated Transfromers Driven Multi-Class Bone Fracture Classification')
title.config(bg='powder blue', fg='olive')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(relx=0.5, y=5, anchor='n')  # centers at top


font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="DeiT Feature Extraction", command=DeiT_Feature_Extraction)
processButton.place(x=20,y=150)
processButton.config(font=ff)

mlpButton = Button(main, text="Dataset Splitting", command=Train_Test_split)
mlpButton.place(x=20,y=200)
mlpButton.config(font=ff)

mlpButton = Button(main, text="Ridge Classifier", command=Existing_Ridge)
mlpButton.place(x=20,y=250)
mlpButton.config(font=ff)

mlpButton = Button(main, text="PAC Classifier", command=Existing_PAC)
mlpButton.place(x=20,y=300)
mlpButton.config(font=ff)

mlpButton = Button(main, text="NCC Classifier", command=Existing_NCC)
mlpButton.place(x=20,y=350)
mlpButton.config(font=ff)


modelButton = Button(main, text="FIGS Classifier", command=Proposed_FIGS)
modelButton.place(x=20,y=400)
modelButton.config(font=ff)


predictButton = Button(main, text="Prediction from Test Image", command=predict)
predictButton.place(x=150,y=400)
predictButton.config(font=ff)

graphButton = Button(main, text="Activate Telegram Bot", command=Telegram_Bot)
graphButton.place(x=150,y=100)
graphButton.config(font=ff)



font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=55)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=350,y=100)
text.config(font=font1)

main.config(bg = 'DarkOrchid3')
main.mainloop()


