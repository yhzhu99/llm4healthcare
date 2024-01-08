import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        
        # Calculate attention scores
        attn_scores = torch.matmul(hidden, encoder_outputs.transpose(1, 2))  
        # attn_scores: [batch_size, 1, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        
        attn_probs = F.softmax(attn_scores, dim=2)  
        # attn_probs: [batch_size, 1, seq_len]
        
        context = torch.matmul(attn_probs, encoder_outputs)  
        # context: [batch_size, 1, hidden_dim]
        
        return context, attn_probs


class GRUAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        self.attention = Attention(hidden_dim)
        self.out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, mask, **kwargs):
        # Pass through GRU
        encoder_outputs, hidden = self.gru(x)
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # hidden: [1, batch_size, hidden_dim]

        # Calculate context vector using attention mechanism
        context, _ = self.attention(hidden.transpose(0, 1), encoder_outputs, mask.unsqueeze(1))
        
        # Concatenate context with hidden state to get the output
        combined = torch.cat((context, hidden.transpose(0, 1)), dim=2)
        output = self.out(combined).squeeze(1)
        return output

