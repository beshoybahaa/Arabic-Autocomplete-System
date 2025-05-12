import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import tkinter as tk
from tkinter import ttk, scrolledtext
import json
import os

class ModelEvaluator:
    def __init__(self, model_dir: str = 'model', tokenizer_dir: str = 'tokenizer'):
        """
        Initialize the model evaluator
        Args:
            model_dir: Directory containing the saved model
            tokenizer_dir: Directory containing the saved tokenizer
        """
        # Initialize GPU device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print("Loading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        
        # Move model to GPU
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def predict_next_word(self, sentence: str, num_predictions: int = 4) -> list:
        """
        Predict the next word given a sentence
        Args:
            sentence: Input sentence
            num_predictions: Number of predictions to return
        Returns:
            List of tuples containing (predicted_word, probability)
        """
        with torch.no_grad():
            # Add BOS token to input
            input_text = f"[BOS] {sentence}"
            inputs = self.tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Get logits for the last position only
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False
            )
            last_token_logits = outputs.logits[0, -1, :]
            
            # Get top k predictions
            top_k_logits, top_k_indices = torch.topk(last_token_logits, num_predictions)
            probs = torch.softmax(top_k_logits, dim=-1)
            
            # Convert to list of predictions
            predictions = []
            for idx, (token_id, prob) in enumerate(zip(top_k_indices, probs)):
                word = self.tokenizer.decode([token_id])
                predictions.append((word, prob.item()))
            
            return predictions

class PredictionGUI:
    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        self.window = tk.Tk()
        self.window.title("Arabic GPT Next Word Predictor")
        self.window.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input text area
        ttk.Label(main_frame, text="Enter your Arabic sentence:").grid(row=0, column=0, sticky=tk.W)
        self.input_text = scrolledtext.ScrolledText(main_frame, width=70, height=5)
        self.input_text.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Predict button
        self.predict_button = ttk.Button(main_frame, text="Predict Next Word", command=self.make_prediction)
        self.predict_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Predictions frame
        predictions_frame = ttk.LabelFrame(main_frame, text="Predictions", padding="10")
        predictions_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Create prediction labels
        self.prediction_labels = []
        for i in range(4):
            label = ttk.Label(predictions_frame, text="")
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            self.prediction_labels.append(label)
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
    def make_prediction(self):
        # Get input text
        input_sentence = self.input_text.get("1.0", tk.END).strip()
        if not input_sentence:
            return
            
        # Get predictions
        predictions = self.evaluator.predict_next_word(input_sentence)
        
        # Update prediction labels
        for i, (word, prob) in enumerate(predictions):
            self.prediction_labels[i].config(
                text=f"Option {i+1}: {word} ({prob*100:.2f}%)"
            )
    
    def run(self):
        self.window.mainloop()

def main():
    # Initialize the evaluator
    evaluator = ModelEvaluator()
    
    # Create and run the GUI
    gui = PredictionGUI(evaluator)
    gui.run()

if __name__ == "__main__":
    main() 