import torch
import streamlit as st
from train import train, save_model, train_load

st.cache_resource()
def load_model(model,optimizer):
    """Loads a trained model and optimizer from a checkpoint."""
    
    model_path = 'data/pred_model.pth'
    checkpoint = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load state dictionaries
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")
    return model,optimizer


def predict_next_word(model, tokenizer, text, top_k=3, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Predicts the next word using a trained model."""
    
    model.to(device)  # Move model to correct device

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