import torch
import torch.nn as nn

class AudioBlock(nn.Module):
    """
    AudioBlock is a block that performs multi-query attention, cross-attention, and feedforward operations on input tensors.

    Args:
        dim (int): The dimension of the input tensors.
        depth (int): The number of times the operations are applied.
        heads (int): The number of attention heads.
        dropout (float, optional): The dropout probability. Defaults to 0.1.

    Attributes:
        dim (int): The dimension of the input tensors.
        depth (int): The number of times the operations are applied.
        heads (int): The number of attention heads.
        dropout (float): The dropout probability.
        attn (MultiQueryAttention): The multi-query attention module.
        cross_attn (CrossAttention): The cross-attention module.
        feedforward (SimpleFeedForward): The feedforward module.

    Methods:
        forward(x: Tensor, audio: Tensor) -&gt; Tensor:
            Performs the forward pass of the AudioBlock module.

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super(AudioBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout

        # Create a list of layers
        self.self_attn_layers = nn.ModuleList([])
        self.cross_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])

        # Add the attention, cross-attention, and feedforward layers to the list
        for _ in range(depth):
            # Add the multi-query attention layer
            self.self_attn_layers.append(
                MultiQueryAttention(dim, heads, *args, **kwargs)
            )
            # Add the cross-attention layer
            self.cross_attn_layers.append(
                CrossAttention(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    *args,
                    **kwargs,
                )
            )
            # Add the feedforward layer
            self.ffn_layers.append(
                SimpleFeedForward(
                    dim, dim * 4, dropout, *args, **kwargs
                )
            )

    def forward(self, x, audio):
        for i in range(self.depth):
            x = self.self_attn_layers[i](x) + x
            x = self.cross_attn_layers[i](x, audio, audio) + x
            x = self.ffn_layers[i](x) + x
        return x

class MultiQueryAttention(nn.Module):
    def __init__(self, dim, heads):
        super(MultiQueryAttention, self).__init__()
        self.multi_head_attn = nn.MultiheadAttention(dim, heads)

    def forward(self, x):
        attn_output, _ = self.multi_head_attn(x, x, x)
        return attn_output


class CrossAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self, x_1, x_2):           # x_2 is new
        queries_1 = x_1 @ self.W_query
        
        keys_2 = x_2 @ self.W_key          # new
        values_2 = x_2 @ self.W_value      # new
        
        attn_scores = queries_1 @ keys_2.T # new 
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1)
        
        context_vec = attn_weights @ values_2
        return context_vec

class SimpleFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(SimpleFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


