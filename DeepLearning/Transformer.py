# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : Transformer.py
@Time    : 2023/9/25 20:15
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of Transformer
-----------------------------------------------------------------------------"""
from typing import List, Union

import torch
from torch import nn, Tensor


class SelfAttention(nn.Module):
    """
    self-attention
    """

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.x2qkv = nn.LazyLinear(d_model * 3)
        self.softmax = nn.Softmax(dim=1)
        self.attn = None

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # convert the input x to query, key and value
        qkv = self.x2qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # print(q.shape, k.shape, v.shape)

        # obtain attention
        attn = torch.einsum("bik,bjk -> bij", q, k) / k.shape[-1]

        # mask the information that should not be seen by now
        if mask:
            # e^(-1e9) will make value near 0
            attn = attn.masked_fill(mask == 0, -1e9)

        # attn softmax score
        # for safe softmax, we minus the maximum in score
        self.attn = self.softmax(attn - attn.max())

        # obtain outputs
        out = torch.einsum("bij, bjk -> bik", self.attn, v)
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    multi-head self-attention
    """

    def __init__(self, d_model: int, nheads: int = 8) -> None:
        super().__init__()

        assert d_model % nheads == 0

        self.x2qkv = nn.LazyLinear(d_model * 3)
        self.softmax = nn.Softmax(dim=1)
        self.nheads = nheads
        self.dk = d_model // nheads
        self.fc = nn.LazyLinear(d_model)
        self.attn = None

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # project x to qkv space
        qkv = self.x2qkv(x)

        # reshape qkv to multi heads
        qkv_nheads = qkv.reshape(qkv.shape[0], qkv.shape[1], self.nheads, self.dk * 3)

        # obtain multi-head q, k and v
        q, k, v = torch.chunk(qkv_nheads, 3, dim=-1)
        # print(q.shape, k.shape, v.shape)

        # get attention
        attn = torch.einsum("bink, bjnk -> bijn", q, k) / (self.dk ** (0.5))

        if mask:
            attn = attn.mased_fill(mask == 0, -1e9)
        self.attn = self.softmax(attn - attn.max())

        # get outputs
        out = torch.einsum("bijn, bjnk -> bink", self.attn, v)

        out = attn.reshape(out.shape[0], out.shape[1], -1)

        # project the outputs back
        out = self.fc(out)

        return out


class MultiHeadAttention(nn.Module):
    """
    multi-head attention. Note that this is not mult-head self-attention.
    This class can be used both for self-attention (encoder) and cross-attention (decoder)
    """

    def __init__(self, d_model: int, nheads: int) -> None:
        super().__init__()
        assert d_model % nheads == 0
        self.softmax = nn.Softmax(dim=1)
        # self.nheads = nheads
        self.dk = d_model // nheads
        self.fc = nn.LazyLinear(d_model)
        self.attn = None

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        # print(q.shape, k.shape, v.shape)
        # get attention
        attn = torch.einsum("bink, bjnk -> bijn", q, k) / (self.dk ** (0.5))
        if mask:
            attn = attn.mased_fill(mask == 0, -1e9)
        self.attn = self.softmax(attn - attn.max())
        # get outputs
        out = torch.einsum("bijn, bjnk -> bink", self.attn, v)

        out = attn.reshape(out.shape[0], out.shape[1], -1)

        # project the outputs back
        out = self.fc(out)
        return out


class MHABlock(nn.Module):
    """
    Multi-head attention with Add & Norm
    """

    def __init__(self, d_model: int, nheads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.nheads = nheads
        self.x2qkv = nn.LazyLinear(d_model * 3)
        self.mha = MultiHeadAttention(d_model, nheads)
        self.norm = nn.LayerNorm(d_model)
        self.attn_type = "self"
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Union[List[Tensor], Tensor], mask: Tensor = None) -> Tensor:
        assert type(x) in [Tensor, list]

        # if it is a tensor, then  it is self-attention
        if isinstance(x, Tensor):
            qkv = self.x2qkv(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)

        # else it is cross-attention
        elif isinstance(x, list):
            q, k, v = x
            self.attn_type = "cross"
        else:
            q, k, v = None, None, None

        # reshape query, key and value to multi-head
        q = q.reshape(q.shape[0], q.shape[1], self.nheads, -1)
        k = k.reshape(k.shape[0], k.shape[1], self.nheads, -1)
        v = v.reshape(v.shape[0], v.shape[1], self.nheads, -1)

        out = self.mha(q, k, v, mask)
        out = self.dropout(out)
        return self.norm(x + out)


class FFN(nn.Module):
    """
    position-wise feed forward networks

    d_model * 4 is d_ff. In the original paper, d_ff is 2048, and d_model is 512
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.LazyLinear(d_model * 4),
            nn.ReLU(),
            nn.LazyLinear(d_model),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        out = self.ff(x)
        return self.norm(x + out)


class EncoderLayer(nn.Module):
    """
    Single encoder layer for Transformer encoder
    """

    def __init__(self, d_model: int, nheads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mha = MHABlock(d_model, nheads, dropout)
        self.ffn = FFN(d_model, dropout)

    def forward(self, x: Union[Tensor, List[Tensor]], mask: Tensor = None) -> Tensor:
        x = self.mha(x, mask)
        return self.ffn(x)


class Encoder(nn.Module):
    """
    Encoder of Transformer
    """

    def __init__(self, d_model: int, nheads: int, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nheads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer for Transformer Decoder
    """

    def __init__(self, d_model: int, nheads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mhsa = MHABlock(d_model, nheads, dropout)
        self.mhca = MHABlock(d_model, nheads, dropout)
        self.ffn = FFN(d_model, dropout)

    def forward(self, x: Union[Tensor, List[Tensor]], mask: Tensor = None) -> Tensor:
        x = self.mhsa(x, mask)
        x = self.mhca(x, mask)
        x = self.ffn(x)
        return x


class Decoder(nn.Module):
    """
    Decoder of Transformer
    """

    def __init__(self, d_model: int, nheads: int, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nheads, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: Union[Tensor, List[Tensor]], mask: Tensor = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PositionalEncoding(nn.Module):
    """
    positional encoding for input and output

    note that embeding is not necessary needed for numerical inputs
    such as remote sensing datasets, while the positional encoding is
    required for any input datasets. Different encoding types, however,
    can be choosen.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # get the position indexes
        pos = torch.arange(max_len).float().reshape(-1, 1)  # [max_len, 1]
        power = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        # print(pos.shape, power.shape)
        theta = pos / power  # broadcating here
        pe = torch.zeros((max_len, d_model)).float()
        pe[:, 0::2] = torch.sin(theta)
        pe[:, 1::2] = torch.cos(theta)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # batch size first, the second term is the seq_len
        # only select the seq_len from max_len
        # print(x.shape, self.pe[:x.shape[1], :].shape)
        x = x + self.pe[:x.shape[1], :]
        return self.dropout(x)


class Transformer(nn.Module):
    """
    Tranformer
    """

    def __init__(self, d_model: int, nheads: int, num_classes: int, num_layers: int = 6, max_len: int = 5000,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.pe_input = PositionalEncoding(d_model, max_len, dropout)
        self.pe_output = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(d_model, nheads, num_layers, dropout)
        self.decoder = Decoder(d_model, nheads, num_layers, dropout)

        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x: Tensor, y: Tensor, x_mask: Tensor = None, y_mask: Tensor = None) -> Tensor:
        # encoder
        x = self.pe_input(x)
        x = self.encoder(x, x_mask)

        # decoder
        x = self.decoder([x, x, y], y_mask)

        # out
        return self.fc(x)


if __name__ == "__main__":
    x_test = torch.randn(8, 20, 512).to("cuda:0")
    f = Decoder(512, 8, 6).to("cuda:0")
    print(f)
    y = f(x_test)
    print(x_test.shape, y.shape)
    f = PositionalEncoding(512, 5000).to("cuda:0")
    y = f(x_test)
    print(x_test.shape, y.shape)
