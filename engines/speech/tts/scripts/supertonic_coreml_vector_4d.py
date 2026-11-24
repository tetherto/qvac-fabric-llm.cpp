"""Channel-first 4D Supertonic vector field; same external protocol as the 3D field."""

import torch
from torch import nn
from torch.nn import functional as F


class Pointwise(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.bias = getattr(source, "bias", None)

    def forward(self, x):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)


class ChannelNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight, self.bias = source.weight, source.bias

    def forward(self, x):
        centered = x - x.mean(dim=1, keepdim=True)
        variance = (centered * centered).mean(dim=1, keepdim=True)
        return (
            centered * torch.rsqrt(variance + 1e-6) * self.weight[None, :, None, None]
            + self.bias[None, :, None, None]
        )


class ConvNeXt(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.kernel, self.bias, self.gamma = source.kernel, source.bias, source.gamma
        self.pad, self.dilation = source.pad, source.dilation
        self.norm = ChannelNorm(source.norm)
        self.pw1, self.pw2 = Pointwise(source.pw1), Pointwise(source.pw2)

    def forward(self, x, valid, last):
        edge = (x * last).sum(dim=3, keepdim=True)
        y = x * valid + edge * (1.0 - valid)
        y = F.conv2d(
            F.pad(y, (self.pad, self.pad, 0, 0), mode="replicate"),
            self.kernel[:, :, None, :],
            self.bias,
            dilation=(1, self.dilation),
            groups=self.kernel.shape[0],
        )
        y = self.pw2(F.gelu(self.pw1(self.norm(y)), approximate="none"))
        return x + self.gamma[None, :, None, None] * y


class TextAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.heads = source.heads
        for name in ("W_query", "W_key", "W_value", "out_fc"):
            setattr(self, name, Pointwise(getattr(source, name)))

    @staticmethod
    def rotate(y, cos, sin):
        cos, sin = cos.transpose(-1, -2), sin.transpose(-1, -2)
        a, b = y[:, :, :32, :], y[:, :, 32:, :]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=2)

    def forward(self, x, text, key_mask, cq, sq, ck, sk):
        batch, _, _, length = x.shape
        text_length = text.shape[-1]
        q = self.rotate(self.W_query(x).reshape(batch, self.heads, 64, length), cq, sq)
        k = self.rotate(
            self.W_key(text).reshape(batch, self.heads, 64, text_length), ck, sk
        )
        v = self.W_value(text).reshape(batch, self.heads, 64, text_length)
        probabilities = torch.softmax(q.transpose(-1, -2) @ k / 16.0 + key_mask, dim=-1)
        y = (v @ probabilities.transpose(-1, -2)).reshape(
            batch, self.heads * 64, 1, length
        )
        return self.out_fc(y)


class StyleAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        for name in ("W_query", "W_key", "W_value", "out_fc"):
            setattr(self, name, Pointwise(getattr(source, name)))

    def forward(self, x, key, value):
        batch, _, _, length = x.shape
        q = self.W_query(x).reshape(batch, 2, 128, length)
        k = torch.tanh(self.W_key(key)).reshape(batch, 2, 128, 50)
        v = self.W_value(value).reshape(batch, 2, 128, 50)
        probabilities = torch.softmax(q.transpose(-1, -2) @ k / 16.0, dim=-1)
        y = (v @ probabilities.transpose(-1, -2)).reshape(batch, 256, 1, length)
        return self.out_fc(y)


class Group(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.first = nn.ModuleList([ConvNeXt(b) for b in source.first])
        self.time = Pointwise(source.time)
        self.mid, self.end = ConvNeXt(source.mid), ConvNeXt(source.end)
        self.text_attn, self.style_attn = (
            TextAttention(source.text_attn),
            StyleAttention(source.style_attn),
        )
        self.text_norm, self.style_norm = (
            ChannelNorm(source.text_norm),
            ChannelNorm(source.style_norm),
        )

    def forward(self, x, text, key, style, time, valid, last, key_mask, cq, sq, ck, sk):
        for block in self.first:
            x = block(x, valid, last)
        x = self.mid(x + self.time(time), valid, last)
        x = self.text_norm(x + self.text_attn(x, text, key_mask, cq, sq, ck, sk))
        x = self.end(x, valid, last)
        return self.style_norm(x + self.style_attn(x, key, style))


class VectorEstimator4D(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.cfg, self.cond_scale, self.uncond_scale = (
            source.cfg,
            source.cond_scale,
            source.uncond_scale,
        )
        self.proj_in, self.proj_out = (
            Pointwise(source.proj_in),
            Pointwise(source.proj_out),
        )
        self.groups = nn.ModuleList([Group(g) for g in source.groups])
        self.tail = nn.ModuleList([ConvNeXt(b) for b in source.tail])
        self.style_key = source.style_key
        if self.cfg:
            self.null_text = source.null_text
            self.null_style_key, self.null_style_value = (
                source.null_style_key,
                source.null_style_value,
            )

    def velocity(
        self,
        noisy_latent,
        text_emb,
        style_ttl,
        latent_mask,
        text_mask,
        time_emb,
        cos_q,
        sin_q,
        cos_k,
        sin_k,
    ):
        valid = latent_mask.unsqueeze(2)
        last = valid - F.pad(valid[..., 1:], (0, 1))
        key_mask = (1.0 - text_mask[:, None, :, :]) * -10000.0
        text = text_emb.unsqueeze(2)
        style = style_ttl.transpose(1, 2).unsqueeze(2)
        key = self.style_key.transpose(1, 2).unsqueeze(2)
        x = noisy_latent.unsqueeze(2)
        if self.cfg:
            x = torch.cat((x, x), dim=0)
            text = torch.cat(
                (
                    text,
                    self.null_text.transpose(1, 2)
                    .unsqueeze(2)
                    .expand(1, 256, 1, text.shape[-1]),
                ),
                dim=0,
            )
            style = torch.cat(
                (style, self.null_style_value.transpose(1, 2).unsqueeze(2)), dim=0
            )
            key = torch.cat(
                (key, self.null_style_key.transpose(1, 2).unsqueeze(2)), dim=0
            )
        time = time_emb[:, :, None, None]
        x = self.proj_in(x) * valid
        for group in self.groups:
            x = group(
                x,
                text,
                key,
                style,
                time,
                valid,
                last,
                key_mask,
                cos_q,
                sin_q,
                cos_k,
                sin_k,
            )
        for block in self.tail:
            x = block(x, valid, last)
        return self.proj_out(x).squeeze(2)

    def forward(self, *inputs):
        velocity = self.velocity(*inputs[:-1])
        if self.cfg:
            velocity = self.cond_scale * velocity[:1] - self.uncond_scale * velocity[1:]
        return inputs[0] + velocity * inputs[3] * inputs[-1]
