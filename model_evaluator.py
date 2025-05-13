import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import tkinter as tk
from tkinter import ttk, scrolledtext
import json
import os
import numpy as np
from typing import List, Tuple, Dict
import re
import pandas as pd

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
        
        # Initialize evaluation metrics
        self.metrics = {
            'perplexity': 0.0
        }
        
        # Load the last row from Arabic_news.csv
        try:
            df = pd.read_csv('Arabic_news.csv')
            all_sentences = []
            
            # Get sentences from last 3 rows
            for i in range(3):
                if len(df) - i - 1 >= 0:  # Check if row exists
                    row_text = df['text'].iloc[-(i+1)]
                    # Split text by periods and clean sentences
                    row_sentences = [s.strip() for s in row_text.split('.') if s.strip()]
                    all_sentences.extend(row_sentences)
            
            # Take first 20 sentences
            all_sentences = all_sentences[:20]
            
            # Create test sentences by removing last word
            self.test_sentences = []
            self.reference_sentences = []
            
            for sentence in all_sentences:
                words = sentence.split()
                if len(words) > 1:  # Only process sentences with more than one word
                    # Remove last word for test sentence
                    test_sentence = ' '.join(words[:-1])
                    self.test_sentences.append(test_sentence)
                    self.reference_sentences.append(sentence)
            
            print(f"Loaded {len(self.test_sentences)} test-reference pairs")
            print("\nExample pairs:")
            for i, (test, ref) in enumerate(zip(self.test_sentences[:3], self.reference_sentences[:3])):
                print(f"\nPair {i+1}:")
                print(f"Test: {test}")
                print(f"Reference: {ref}")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            # Fallback to default sentences
            self.test_sentences = ["مرحبا كيف"]
            self.reference_sentences = ["مرحبا كيف حالك"]
    
    def calculate_perplexity(self, test_sentences: List[str]) -> float:
        """
        Calculate perplexity on a set of test sentences
        Args:
            test_sentences: List of test sentences
        Returns:
            Perplexity score
        """
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for sentence in test_sentences:
                # Add BOS token to input
                input_text = f"[BOS] {sentence}"
                inputs = self.tokenizer(input_text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                
                # Get model outputs
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                # Accumulate loss and token count
                total_loss += outputs.loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
    
    def evaluate_model(self, test_sentences: List[str]) -> Dict[str, float]:
        """
        Evaluate the model using perplexity
        Args:
            test_sentences: List of test sentences
        Returns:
            Dictionary containing evaluation metrics
        """
        perplexity = self.calculate_perplexity(test_sentences)
        
        self.metrics = {
            'perplexity': perplexity
        }
        
        return self.metrics
    
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
        self.window.geometry("800x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input text area with RTL support
        ttk.Label(main_frame, text="Enter your Arabic sentence:").grid(row=0, column=0, sticky=tk.W)
        self.input_text = scrolledtext.ScrolledText(
            main_frame, 
            width=70, 
            height=5,
            font=('Arial', 12)  # Set font size
        )
        self.input_text.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Configure RTL
        self.input_text.tag_configure("rtl", justify="right")
        self.input_text.tag_add("rtl", "1.0", "end")
        
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
        
        # Evaluation metrics frame
        metrics_frame = ttk.LabelFrame(main_frame, text="Model Evaluation Metrics", padding="10")
        metrics_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Create metrics labels
        self.perplexity_label = ttk.Label(metrics_frame, text="Perplexity: N/A")
        self.perplexity_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # Evaluate button
        self.evaluate_button = ttk.Button(main_frame, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_button.grid(row=5, column=0, columnspan=2, pady=10)
        
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
    
    def evaluate_model(self):
        # Use the sentences from the CSV file
        test_sentences = self.evaluator.test_sentences
        
        # Evaluate model
        metrics = self.evaluator.evaluate_model(test_sentences)
        
        # Update metrics labels
        self.perplexity_label.config(text=f"Perplexity: {metrics['perplexity']:.2f}")
        
        # Display number of sentences used
        print(f"Evaluated using {len(test_sentences)} test sentences")
    
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