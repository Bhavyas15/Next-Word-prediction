import torch
import streamlit as st
from train import train, save_model, train_load
import gdown
import os

def download_model():
    # Google Drive file ID (Extracted from the link)
    file_id = "1bZ9XNC5UmONE1WHwL632eJ1WjCC3g6w2"

    # Destination path
    destination = "data/pred_model.pth"

    # Download the file
    output=gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)

    # Rename the file if needed
    if output and output != destination:
        os.rename(output, destination)

    print("✅ Model downloaded successfully!")

st.cache_resource()
def load_model(model,optimizer):
    """Loads a trained model and optimizer from a checkpoint."""
    
    model_path = 'data/pred_model.pth'
    checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=False)

    # Load state dictionaries
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")
    return model,optimizer


def predict_next_word(model, tokenizer, text, top_k=3, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Predicts the next word using a trained model."""
    
    # model.to(device)  # Move model to correct device

    # Tokenize input and convert to tensor
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    m = input_ids==0
    m = m.float().masked_fill(m == 1, float('-inf'))

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids,m)

    # Get the predicted token (this depends on your model's output format)
    logits = outputs[:, :-1, :]
    prediction_ids = torch.argmax(outputs, axis=-1)
    
    top_k_ids = torch.topk(logits, top_k, dim=-1).indices[0]
    print(top_k_ids)
    predicted_words = [tokenizer.decode(token_id) for token_id in top_k_ids]
    print(predicted_words)

    # Convert token ID back to word
    predicted_word = tokenizer.decode(prediction_ids[0])

    return predicted_word,input_ids,predicted_words

def space(n):
    for _ in range(n): 
        st.write("")  # blank line
    return

def initialize_model(model,optimizer,model_pth, load_dataset, tokenizer, criterion):
    if not model_pth:
        common_allen = load_dataset("allenai/common_gen")
        train_loader = train_load(common_allen,tokenizer)
        train_losses = train(model, criterion, optimizer, train_loader, epochs = 50)
        save_model(model,optimizer)
    model, optimizer = load_model(model,optimizer)
    return model,optimizer
