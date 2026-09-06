import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):    
    def __init__(self, sequence_length, vocab_size, output_dim, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, output_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(sequence_length, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        seq_len = inputs.size(1)
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0)
        
        embeddings = self.token_embeddings(inputs) + self.position_embeddings(positions)
        
        padding_mask = inputs == 0
        embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1), 0)
        
        return self.dropout(embeddings)


class TransformerEncoderLayer(nn.Module):    
    def __init__(self, hidden_dim, intermediate_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.self_attention_layernorm = nn.LayerNorm(hidden_dim)
        
        self.feed_forward_1 = nn.Linear(hidden_dim, intermediate_dim)
        self.feed_forward_2 = nn.Linear(intermediate_dim, hidden_dim)
        self.feed_forward_layernorm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, source, source_mask=None):
        norm_source = self.self_attention_layernorm(source)
        attn_out, _ = self.self_attention(
            query=norm_source,
            key=norm_source,
            value=norm_source,
            key_padding_mask=source_mask
        )
        x = source + self.dropout(attn_out)
        
        norm_x = self.feed_forward_layernorm(x)
        ffn_out = self.feed_forward_2(self.dropout(F.gelu(self.feed_forward_1(norm_x))))
        x = x + self.dropout(ffn_out)
        return x


class TransformerDecoderLayer(nn.Module):    
    def __init__(self, hidden_dim, intermediate_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.self_attention_layernorm = nn.LayerNorm(hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.cross_attention_layernorm = nn.LayerNorm(hidden_dim)
        
        self.feed_forward_1 = nn.Linear(hidden_dim, intermediate_dim)
        self.feed_forward_2 = nn.Linear(intermediate_dim, hidden_dim)
        self.feed_forward_layernorm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, source, source_mask=None, target_mask=None):
        tgt_len = target.size(1)
        
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=target.device, dtype=torch.bool),
            diagonal=1
        )
        
        norm_target = self.self_attention_layernorm(target)
        attn_out, _ = self.self_attention(
            query=norm_target,
            key=norm_target,
            value=norm_target,
            attn_mask=causal_mask,
            key_padding_mask=target_mask
        )
        x = target + self.dropout(attn_out)
        
        norm_x = self.cross_attention_layernorm(x)
        cross_out, _ = self.cross_attention(
            query=norm_x,
            key=source,
            value=source,
            key_padding_mask=source_mask
        )
        x = x + self.dropout(cross_out)
        
        norm_x = self.feed_forward_layernorm(x)
        ffn_out = self.feed_forward_2(self.dropout(F.gelu(self.feed_forward_1(norm_x))))
        x = x + self.dropout(ffn_out)
        return x


class TransformerModel(nn.Module):    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        sequence_length,
        hidden_dim=256,
        intermediate_dim=1024,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()
        
        self.source_embedding = PositionalEmbedding(
            sequence_length, src_vocab_size, hidden_dim, dropout
        )
        self.target_embedding = PositionalEmbedding(
            sequence_length, tgt_vocab_size, hidden_dim, dropout
        )
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, intermediate_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, intermediate_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.encoder_final_norm = nn.LayerNorm(hidden_dim)
        self.decoder_final_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(hidden_dim, tgt_vocab_size, bias=False)
        self.classifier.weight = self.target_embedding.token_embeddings.weight

    def forward(self, source, target):
        source_mask = (source == 0)
        target_mask = (target == 0)
        
        x_enc = self.source_embedding(source)
        for layer in self.encoder_layers:
            x_enc = layer(x_enc, source_mask=source_mask)
        x_enc = self.encoder_final_norm(x_enc)
        
        x_dec = self.target_embedding(target)
        for layer in self.decoder_layers:
            x_dec = layer(
                x_dec,
                source=x_enc,
                source_mask=source_mask,
                target_mask=target_mask
            )
        x_dec = self.decoder_final_norm(x_dec)
        
        logits = self.classifier(self.dropout(x_dec))
        return logits
