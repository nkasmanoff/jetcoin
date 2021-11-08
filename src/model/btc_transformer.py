"""
Following some adjustments, now have a transformer decoder model which
accepts as input a sequence of prices, and returns a predicted value based on the
hidden representation of the last token.

Does not use Masked MHA for all intermediate tokens, so I would avoid that. For that reason
this is probably the same as an encoder?
"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)




def create_self_mask(target_len, device=None):
    # Prevent leftward information flow in self-attention (of decoder)
    ones = torch.ones(target_len, target_len, dtype=torch.uint8,
                      device=device)
    t_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)

    return t_self_mask


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate,output_size=None):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        if output_size:
            self.layer2 = nn.Linear(filter_size, output_size)
        else:
            self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=4):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask, cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x.masked_fill_(mask.unsqueeze(1), -1e9)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x



class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, self_mask):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, self_mask)
        y = self.self_attention_dropout(y)
        x = x + y

       # if enc_output is not None:
       #     y = self.enc_dec_attention_norm(x)
        #    y = self.enc_dec_attention(y, enc_output, enc_output, i_mask,
                                   #    cache)
          #  y = self.enc_dec_attention_dropout(y)
        #   x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x



class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x, self_mask):
        decoder_output = x
        for i, dec_layer in enumerate(self.layers):
            decoder_output = dec_layer(decoder_output,
                                       self_mask)
        return self.last_norm(decoder_output)





class Transformer(nn.Module):
    def __init__(self,
                 n_layers=6,
                 hidden_size=32,
                 filter_size=64,
                 dropout_rate=0.1,
                window_size=32):
        super(Transformer, self).__init__()

        self.window_size = window_size
        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5



        self.price_embedding = nn.Linear(window_size,hidden_size) # this 1 can be scaled to # of cryptos, trends, etc. Need to fix loaders as well for that but interesting opportunity.

        nn.init.normal_(self.price_embedding.weight, mean=0,
                        std=hidden_size**-0.5)

        self.decoder = Decoder(hidden_size, filter_size,
                               dropout_rate, n_layers)

        self.output_mlp = FeedForwardNetwork(hidden_size,filter_size,dropout_rate,output_size=1)


        # For positional encoding
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)



    def forward(self, x):

        self_mask = create_self_mask(self.window_size)
        x = self.decode(x, self_mask)
        x = self.output_mlp(x[:,-1,:])
        return x


    def decode(self, x, self_mask):



        x = torch.stack([x[i,:] * torch.eye(x.shape[1]) for i in range(x.shape[0])]) # my way of making hidden size for this "embedding"
        x *= self.emb_scale
        x += self.get_position_encoding(x)

        # decoder
        output = self.decoder(x, self_mask)

        return output # and take the last token of this..

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal


# TODO - pytests

# expected output shape is probably what I'd want to see.
