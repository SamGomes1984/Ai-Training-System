import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to load and interact with model
def interact_with_model(model_name, input_text):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Generate a response
    outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Gradio interface
def create_gradio_interface():
    model_names = ["microsoft/Phi-3-mini-4k-instruct", "gpt2"]  # Example models
    
    # Gradio interface setup
    interface = gr.Interface(
        fn=interact_with_model,  # Function to call
        inputs=[
            gr.Dropdown(choices=model_names, label="Select Model"),  # Dropdown to select model
            gr.Textbox(label="Enter Text", placeholder="Type something...")  # Text input for text
        ],
        outputs="text",  # Output text area
        live=True  # Enable live updates
    )

    # Launch the interface with a public URL
    interface.launch(share=True)

# Run the Gradio interface
if __name__ == "__main__":
    create_gradio_interface()
