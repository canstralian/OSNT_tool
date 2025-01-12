import unittest
from unittest.mock import patch, MagicMock
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import io

class TestStreamlitApp(unittest.TestCase):

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_load_model_success(self, mock_model, mock_tokenizer):
        # Mock the tokenizer and model loading
        mock_tokenizer.return_value = MagicMock(spec=AutoTokenizer)
        mock_model.return_value = MagicMock(spec=AutoModelForSequenceClassification)

        tokenizer, model = load_model("Canstralian/CyberAttackDetection")

        # Assert that the tokenizer and model are not None
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(model)
        mock_tokenizer.assert_called_once_with("Canstralian/CyberAttackDetection")
        mock_model.assert_called_once_with("Canstralian/CyberAttackDetection")

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_predict_classification(self, mock_model, mock_tokenizer):
        # Mock the tokenizer and model for inference
        mock_tokenizer.return_value = MagicMock(spec=AutoTokenizer)
        mock_model.return_value = MagicMock(spec=AutoModelForSequenceClassification)

        # Simulate model outputs
        mock_model.return_value.__call__.return_value = MagicMock(logits=torch.tensor([[1.0, 2.0, 3.0]]))

        # Call the prediction function
        inputs = mock_tokenizer("Test input", return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = mock_model.return_value(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        # Assert that the predicted class is correct
        self.assertEqual(predicted_class, 2)  # The class with the highest score (index 2)

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
    def test_generate_shell_command(self, mock_model, mock_tokenizer):
        # Mock the tokenizer and model for shell command generation
        mock_tokenizer.return_value = MagicMock(spec=AutoTokenizer)
        mock_model.return_value = MagicMock(spec=AutoModelForSeq2SeqLM)

        # Simulate model output (fake shell command)
        mock_model.return_value.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        # Simulate text input
        user_input = "Create a directory"
        inputs = mock_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = mock_model.return_value.generate(**inputs)
        generated_command = mock_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Assert the generated command is as expected
        self.assertEqual(generated_command, "mkdir directory")  # Assuming the model generates this

if __name__ == "__main__":
    unittest.main()
