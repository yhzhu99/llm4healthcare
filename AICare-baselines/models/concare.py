import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.utils import generate_mask, get_last_visit


class FinalAttentionQKV(nn.Module):
    def __init__(
        self,
        attention_input_dim: int,
        attention_hidden_dim: int,
        attention_type: str = "add",
        dropout: float = 0.5,
    ):
        super(FinalAttentionQKV, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(
            torch.zeros(
                1,
            )
        )
        self.b_out = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(
            torch.randn(2 * attention_input_dim, attention_hidden_dim)
        )
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        (
            batch_size,
            time_step,
            lab_dim,
        ) = x.size()  # batch_size * lab_dim + 1 * hidden_dim(i)
        input_q = self.W_q(x[:, -1, :])  # b h
        input_k = self.W_k(x)  # b t h
        input_v = self.W_v(x)  # b t h

        if self.attention_type == "add":  # B*T*I  @ H*I

            q = torch.reshape(
                input_q, (batch_size, 1, self.attention_hidden_dim)
            )  # B*1*H
            h = q + input_k + self.b_in  # b t h
            h = self.tanh(h)  # B*T*H
            e = self.W_out(h)  # b t 1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "mul":
            q = torch.reshape(
                input_q, (batch_size, self.attention_hidden_dim, 1)
            )  # B*h 1
            e = torch.matmul(input_k, q).squeeze()  # b t

        elif self.attention_type == "concat":
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        else:
            raise ValueError(
                "Unknown attention type: {}, please use add, mul, concat".format(
                    self.attention_type
                )
            )

        a = self.softmax(e)  # B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()  # B*I

        return v, a


class PositionwiseFeedForward(nn.Module):  # new added
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x)))), None


class PositionalEncoding(nn.Module):  # new added / not use anymore
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pos = self.pe[:, : x.size(1)].clone().requires_grad_(False)
        x = x + pos
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, self.d_k * self.h) for _ in range(3)]
        )
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)  # b h t d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # b h t t
        if mask is not None:  # 1 1 t t
            scores = scores.masked_fill(mask == 0, -1e9)  # b h t t 下三角
        p_attn = torch.softmax(scores, dim=-1)  # b h t t
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  # b h t v (d_k)

    def cov(self, m, y=None):
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 1 1 t t

        nbatches = query.size(0)  # b
        lab_dim = query.size(1)  # i+1
        feature_dim = query.size(1)  # i+1

        # x size -> # batch_size * d_input * hidden_dim

        # d_model => h * d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]  # b num_head d_input d_k

        x, self.attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )  # b num_head d_input d_v (d_k)

        x = (
            x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        )  # batch_size * d_input * hidden_dim

        # DeCov
        DeCov_contexts = x.transpose(0, 1).transpose(1, 2)  # I+1 H B
        Covs = self.cov(DeCov_contexts[0, :, :])
        DeCov_loss = 0.5 * (
            torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2
        )
        for i in range(feature_dim - 1):
            Covs = self.cov(DeCov_contexts[i + 1, :, :])
            DeCov_loss += 0.5 * (
                torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2
            )

        return self.final_linear(x), DeCov_loss


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]), returned_value[1]


class SingleAttention(nn.Module):
    def __init__(
        self,
        attention_input_dim,
        attention_hidden_dim,
        attention_type="add",
        time_aware=False,
    ):
        super(SingleAttention, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.time_aware = time_aware

        # batch_time = torch.arange(0, batch_mask.size()[1], dtype=torch.float32).reshape(1, batch_mask.size()[1], 1)
        # batch_time = batch_time.repeat(batch_mask.size()[0], 1, 1)

        if attention_type == "add":
            if self.time_aware:
                # self.Wx = nn.Parameter(torch.randn(attention_input_dim+1, attention_hidden_dim))
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.bh = nn.Parameter(
                torch.zeros(
                    attention_hidden_dim,
                )
            )
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "mul":
            self.Wa = nn.Parameter(
                torch.randn(attention_input_dim, attention_input_dim)
            )
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "concat":
            if self.time_aware:
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim + 1, attention_hidden_dim)
                )
            else:
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim, attention_hidden_dim)
                )

            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == "new":
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.Wx = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )

            self.rate = nn.Parameter(torch.zeros(1) + 0.8)
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        else:
            raise RuntimeError(
                "Wrong attention type. Please use 'add', 'mul', 'concat' or 'new'."
            )

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, mask, device):

        (
            batch_size,
            time_step,
            lab_dim,
        ) = x.size()  # batch_size * time_step * hidden_dim(i)

        time_decays = (
            torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32)
            .unsqueeze(-1)
            .unsqueeze(0)
            .to(device=device)
        )  # 1*t*1
        b_time_decays = time_decays.repeat(batch_size, 1, 1) + 1  # b t 1

        if self.attention_type == "add":  # B*T*I  @ H*I
            last_visit = get_last_visit(x, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            if self.time_aware == True:
                k = torch.matmul(x, self.Wx)  # b t h
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)  # b t h
            else:
                k = torch.matmul(x, self.Wx)  # b t h
            h = q + k + self.bh  # b t h
            if self.time_aware:
                h += time_hidden
            h = self.tanh(h)  # B*T*H
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        elif self.attention_type == "mul":
            last_visit = get_last_visit(x, mask)
            e = torch.matmul(last_visit, self.Wa)  # b i
            e = (
                torch.matmul(e.unsqueeze(1), x.permute(0, 2, 1)).reshape(
                    batch_size, time_step
                )
                + self.ba
            )  # b t
        elif self.attention_type == "concat":
            last_visit = get_last_visit(x, mask)
            q = last_visit.unsqueeze(1).repeat(1, time_step, 1)  # b t i
            k = x
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            if self.time_aware:
                c = torch.cat((c, b_time_decays), dim=-1)  # B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "new":
            last_visit = get_last_visit(x, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            k = torch.matmul(x, self.Wx)  # b t h
            dot_product = torch.matmul(q, k.transpose(1, 2)).reshape(
                batch_size, time_step
            )  # b t
            denominator = self.sigmoid(self.rate) * (
                torch.log(2.72 + (1 - self.sigmoid(dot_product)))
                * (b_time_decays.reshape(batch_size, time_step))
            )
            e = self.relu(self.sigmoid(dot_product) / (denominator))  # b * t
        else:
            raise ValueError(
                "Wrong attention type. Plase use 'add', 'mul', 'concat' or 'new'."
            )

        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        a = self.softmax(e)  # B*T
        v = torch.matmul(a.unsqueeze(1), x).reshape(batch_size, lab_dim)  # B*I

        return v, a


class ConCareLayer(nn.Module):
    """ConCare layer.

    Paper: Liantao Ma et al. Concare: Personalized clinical feature embedding via capturing the healthcare context. AAAI 2020.

    This layer is used in the ConCare model. But it can also be used as a
    standalone layer.

    Args:
        lab_dim: dynamic feature size.
        demo_dim: static feature size, if 0, then no static feature is used.
        hidden_dim: hidden dimension of the channel-wise GRU, default 128.
        transformer_hidden: hidden dimension of the transformer, default 128.
        num_head: number of heads in the transformer, default 4.
        pe_hidden: hidden dimension of the positional encoding, default 64.
        dropout: dropout rate, default 0.5.

    Examples:
        >>> from pyhealth.models import ConCare
        >>> x = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = ConCare(64)
        >>> c, _ = layer(x)
        >>> c.shape
        torch.Size([3, 128])
    """

    def __init__(
        self,
        lab_dim: int,
        demo_dim: int = 0,
        hidden_dim: int = 128,
        num_head: int = 4,
        pe_hidden: int = 64,
        dropout: int = 0.5,
    ):
        super(ConCareLayer, self).__init__()

        # hyperparameters
        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim  # d_model
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
        # self.output_dim = output_dim
        self.dropout = dropout
        self.demo_dim = demo_dim

        # layers
        self.PositionalEncoding = PositionalEncoding(
            self.transformer_hidden, dropout=0, max_len=400
        )

        self.GRUs = nn.ModuleList(
            [
                nn.GRU(1, self.hidden_dim, batch_first=True)
                for _ in range(self.lab_dim)
            ]
        )
        self.LastStepAttentions = nn.ModuleList(
            [
                SingleAttention(
                    self.hidden_dim,
                    8,
                    attention_type="new",
                    time_aware=True,
                )
                for _ in range(self.lab_dim)
            ]
        )

        self.FinalAttentionQKV = FinalAttentionQKV(
            self.hidden_dim,
            self.hidden_dim,
            attention_type="mul",
            dropout=self.dropout,
        )

        self.MultiHeadedAttention = MultiHeadedAttention(
            self.num_head, self.transformer_hidden, dropout=self.dropout
        )
        self.SublayerConnection = SublayerConnection(
            self.transformer_hidden, dropout=self.dropout
        )

        self.PositionwiseFeedForward = PositionwiseFeedForward(
            self.transformer_hidden, self.pe_hidden, dropout=0.1
        )

        if self.demo_dim > 0:
            self.demo_proj_main = nn.Linear(self.demo_dim, self.hidden_dim)

        self.dropout = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, lab_dim].
            static: a tensor of shape [batch size, demo_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
        """
        # rnn will only apply dropout between layers
        if self.demo_dim > 0:
            demo_main = self.tanh(self.demo_proj_main(static)).unsqueeze(
                1
            )  # b hidden_dim

        batch_size = x.size(0)
        time_step = x.size(1)
        feature_dim = x.size(2)

        if self.transformer_hidden % self.num_head != 0:
            raise ValueError("transformer_hidden must be divisible by num_head")

        # forward
        GRU_embeded_input = self.GRUs[0](
            x[:, :, 0].unsqueeze(-1).to(device=x.device),
            torch.zeros(batch_size, self.hidden_dim)
            .to(device=x.device)
            .unsqueeze(0),
        )[
            0
        ]  # b t h
        Attention_embeded_input = self.LastStepAttentions[0](
            GRU_embeded_input, mask, x.device
        )[0].unsqueeze(
            1
        )  # b 1 h

        for i in range(feature_dim - 1):
            embeded_input = self.GRUs[i + 1](
                x[:, :, i + 1].unsqueeze(-1),
                torch.zeros(batch_size, self.hidden_dim)
                .to(device=x.device)
                .unsqueeze(0),
            )[
                0
            ]  # b 1 h
            embeded_input = self.LastStepAttentions[i + 1](
                embeded_input, mask, x.device
            )[0].unsqueeze(
                1
            )  # b 1 h
            Attention_embeded_input = torch.cat(
                (Attention_embeded_input, embeded_input), 1
            )  # b i h

        if self.demo_dim > 0:
            Attention_embeded_input = torch.cat(
                (Attention_embeded_input, demo_main), 1
            )  # b i+1 h
        posi_input = self.dropout(
            Attention_embeded_input
        )  # batch_size * d_input+1 * hidden_dim

        contexts = self.SublayerConnection(
            posi_input,
            lambda _: self.MultiHeadedAttention(
                posi_input, posi_input, posi_input, None
            ),
        )  # # batch_size * d_input * hidden_dim

        DeCov_loss = contexts[1]
        contexts = contexts[0]

        contexts = self.SublayerConnection(
            contexts, lambda _: self.PositionwiseFeedForward(contexts)
        )[0]

        weighted_contexts, a = self.FinalAttentionQKV(contexts)
        return weighted_contexts, DeCov_loss

class ConCare(nn.Module):
    def __init__(
        self,
        lab_dim: int,
        demo_dim: int = 0,
        hidden_dim: int = 128,
        num_head: int = 4,
        pe_hidden: int = 64,
        dropout: int = 0.0,
        **kwargs
    ):
        super(ConCare, self).__init__()

        # hyperparameters
        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim  # d_model
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
        # self.output_dim = output_dim
        self.dropout = nn.Dropout(p=dropout)
        self.demo_dim = demo_dim
        self.concare_layer = ConCareLayer(lab_dim=lab_dim, demo_dim=demo_dim, hidden_dim=hidden_dim, num_head=num_head, pe_hidden=pe_hidden, dropout=dropout)

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, lab_dim].
            static: a tensor of shape [batch size, demo_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
        """
        # rnn will only apply dropout between layers
        batch_size, time_steps, _ = x.size()
        # out = torch.zeros((batch_size, time_steps, self.hidden_dim))
        # decov_loss = 0
        # for cur_time in range(time_steps):
        #     cur_x = x[:, :cur_time+1, :]
        #     cur_mask = mask[:, :cur_time+1]
        #     cur_out, decov = self.concare_layer(cur_x, static, cur_mask)
        #     out[:, cur_time, :] = cur_out
        #     decov_loss += decov
        # decov_loss /= time_steps
        # out = self.dropout(out)
        # return out, decov_loss
        out, decov_loss = self.concare_layer(x, static, mask)
        decov_loss = decov_loss / time_steps
        out = self.dropout(out)
        return out, decov_loss