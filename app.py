import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch

# Function to handle model loading
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Function to train the model
def train_model(model, tokenizer, data):
    # Tokenize the data
    inputs = tokenizer(data['input'], return_tensors='pt', padding=True, truncation=True)
    labels = tokenizer(data['output'], return_tensors='pt', padding=True, truncation=True).input_ids
    
    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Training loop (simplified)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(3):  # 3 epochs as an example
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")
    
    return model

# Streamlit Interface
def main():
    st.title("AI Model Training System")
    
    model_name = st.selectbox("Select Model", ["gpt2", "bert-base-uncased", "distilbert-base-uncased"])
    
    data_input = st.text_area("Enter Training Data (JSON Format)", '{"input": ["Hello"], "output": ["Hi"]}')
    if st.button("Start Training"):
        if data_input:
            try:
                # Parse the input data
                data = eval(data_input)
                dataset = Dataset.from_dict(data)
                
                # Load and train the model
                model, tokenizer = load_model(model_name)
                trained_model = train_model(model, tokenizer, dataset)
                st.success("Model Training Completed Successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please provide input data.")

if __name__ == "__main__":
    main()
