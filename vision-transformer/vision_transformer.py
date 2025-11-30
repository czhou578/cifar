import torch
import math
from torch import nn

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

config = {
    "patch_size": 4,
    "num_classes": 100,
    "num_channels": 3,
    "num_hidden_layers": 6,
    "hidden_size": 256,
    "image_size": 32,
    "dropout_rate": 0.1,
    "num_attent_heads": 8,
    "intermediate_size": 4 * 256,
    "qkv_bias": True,
    "initializer_range": 0.02
}

class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.image_size = config["image_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)


    def forward(self, x):
        """
        x: (batch_size, num_channels, height, width)
        new_x: (batch_size, num_patches, hidden_size)
        """
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config["hidden_size"])) # Corrected size
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings # Used position_embeddings
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value)

        return (attention_output, attention_probs)

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config["hidden_size"]
        self.num_attent_heads = config["num_attent_heads"]

        self.attent_head_size = self.hidden_size // self.num_attent_heads # Fixed typo
        self.qkv_bias = config["qkv_bias"]
        self.heads = nn.ModuleList([])

        for _ in range(self.num_attent_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attent_head_size,
                config["dropout_rate"],
                self.qkv_bias
            )

            self.heads.append(head)

        self.all_head_size = self.num_attent_heads * self.attent_head_size # Added initialization for all_head_size

        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([attent for attent, _ in attention_outputs], dim=-1)

        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        return attention_output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout_rate"])


    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)

        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(config["hidden_size"])
        self.attention = MultiHeadedAttention(config)

    def forward(self, x):
        attention_input = self.layer_norm1(x)
        attention_output = self.attention(attention_input) # Removed attention_probs as it's not returned by MultiHeadedAttention
        x = x + attention_output # Add skip connection for attention output

        mlp_input = self.layer_norm2(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output # Add skip connection for MLP output

        return x, attention_output # Returning attention_output for consistency if needed later, but only x is used

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)


    def forward(self, x, output_attentions):
        all_attention = []

        for block in self.blocks:
            # The block forward method has been updated to return `x, attention_output`
            # We can capture attention_output if output_attentions is True.
            x, attention_output_for_block = block(x)
            if output_attentions:
                all_attention.append(attention_output_for_block) # Storing attention_output, not attention_probs

        return (x, all_attention)

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)

        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)


    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions)
        logits = self.classifier(encoder_output[:, 0, :])

        if output_attentions:
           return (logits, all_attentions)
        return logits # Return logits if output_attentions is False


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
