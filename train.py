import torch
import torch.nn as nn
import numpy as np

# Check if CUDA is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read text file
with open('new.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create vocabulary and encoding mappings
char = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(char)}
itos = {i: ch for i, ch in enumerate(char)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Define sequence length
seq_len = 50  # Reduced sequence length
sequences = []
targets = []

# Create input sequences and targets
for i in range(0, len(text) - seq_len):
    seq = text[i:i + seq_len]
    target = text[i + seq_len]
    sequences.append(encode(seq))
    targets.append(encode(target)[0])

# Convert to numpy arrays
sequences = np.array(sequences)
targets = np.array(targets)

# Transformer-based model definition
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size, dropout=0.1):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embed_size))  # learnable positional encoding
        
        # Transformer Encoder Layer with batch_first=True
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Fully connected output layer
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        # Get the embeddings for the input sequence
        x = self.embedding(x) + self.positional_encoding  # Add positional encoding to embeddings
        
        # Pass through the transformer
        x = self.transformer(x)
        
        # We are interested in the output of the last time step
        x = x[:, -1, :]  # Get the last token's output
        
        # Pass through the fully connected layer to predict next character
        x = self.fc(x)
        
        return x


# Model parameters
vocab_size = len(char)
embed_size = 64  # You can experiment with different sizes
num_heads = 8  # Number of attention heads
num_layers = 4  # Number of transformer layers
hidden_size = 128  # Hidden size of transformer

# Initialize model and move to device
model = CharTransformer(vocab_size, embed_size, num_heads, num_layers, hidden_size).to(device)
X = torch.tensor(sequences, dtype=torch.int64).to(device)
y = torch.tensor(targets, dtype=torch.int64).to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Create batches
batch_size = 32  # Increased batch size
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop with gradient clipping
epochs = 20  # Number of epochs
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()  # Update learning rate
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

# Text generation function with temperature sampling
def generate_text(seed_text, length=100, temperature=1.0):
    model.eval()
    input_seq = [stoi[c] for c in seed_text]
    input_seq = torch.tensor(input_seq, dtype=torch.int64).unsqueeze(0).to(device)
    
    generated_text = seed_text
    with torch.no_grad():
        for _ in range(length):
            output = model(input_seq)
            output = output / temperature
            probabilities = torch.softmax(output, dim=1)
            predicted_index = torch.multinomial(probabilities, num_samples=1).item()
            predicted_char = itos[predicted_index]
            generated_text += predicted_char
            input_seq = torch.cat((input_seq[:, 1:], torch.tensor([[predicted_index]], dtype=torch.int64).to(device)), dim=1)
    
    return generated_text

# Test text generation
seed_text = "with a laugh that"
generated_text = generate_text(seed_text, 50, temperature=0.8)
print(generated_text)
