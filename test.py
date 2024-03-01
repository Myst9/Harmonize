from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

new_sentence = """You must be a big fool"""
tokenized_sentence = tokenizer.tokenize(new_sentence)

MAX_LEN = 128

# Use the BERT Tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)

# Pad the input tokens
padded_input = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention mask
attention_mask = [float(i > 0) for i in padded_input[0]]

import torch

input_tensor = torch.tensor(padded_input)
attention_mask_tensor = torch.tensor([attention_mask])

from transformers import BertForSequenceClassification

# Load the pre-trained model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the saved model state
model.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(input_tensor, attention_mask=attention_mask_tensor)

# Get the predicted class probabilities
predicted_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get the predicted class index
predicted_class = torch.argmax(predicted_probs, dim=1).item()

print("Predicted Class:", predicted_class)
print("Predicted Probabilities:", predicted_probs.numpy())