import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import warnings

# Suppress specific UserWarnings related to max_length in transformers
warnings.filterwarnings(
    "ignore", message=".*Using the model-agnostic default `max_length`.*"
)


class SimpleModel(nn.Module):
    """
    A simple neural network model that includes an embedding layer
    followed by two blocks, each containing a linear layer and a layer normalization.

    The model takes a tensor of shape (batch_size, sequence_length) as input
    and returns a tensor of shape (batch_size, 2) as output.

    The model is initialized using Kaiming Uniform initialization.
    """

    def __init__(self):
        super(SimpleModel, self).__init__()

        torch.manual_seed(42)

        self.embedding_layer = nn.Embedding(2, 2)

        # First Block
        self.fc1 = nn.Linear(2, 4)
        self.norm1 = nn.LayerNorm(4)

        # Second Block
        self.fc2 = nn.Linear(4, 2)
        self.norm2 = nn.LayerNorm(2)

        self.output_layer = nn.Linear(2, 2)

    def forward(self, x):
        embeddings = self.embedding_layer(x)

        # First Block
        embeddings = self.fc1(embeddings)
        embeddings = self.norm1(embeddings)

        # Second Block
        embeddings = self.fc2(embeddings)
        embeddings = self.norm2(embeddings)

        output = self.output_layer(embeddings)
        return output


def initialize_weights(layer):
    """
    A function that initializes the weights of a layer using Kaiming Uniform initialization.

    Args:
    - layer: The layer to initialize
    """
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.Embedding):
        nn.init.uniform_(layer.weight)


def generate_output(model, processor, image, dtype):
    """
    A function that generates a caption for an image using a model and a tokenizer.

    Args:
    - model: The model to generate the caption
    - processor: The tokenizer to process the image
    - image: The image to generate the caption for
    - dtype: The data type to use for the model
    """
    inputs = processor(image, return_tensors="pt").to(dtype)
    generated_output = model.generate(**inputs)
    return processor.decode(generated_output[0], skip_special_tokens=True)


def fetch_image(image_url):
    """
    A function that fetches an image from a URL and returns it as a PIL image.

    Args:
    - image_url: The URL of the image to fetch
    """
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    return image


def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int8):
    """
    A function that quantizes a tensor using a scale and zero point.

    Args:
    - tensor: The tensor to quantize
    - scale: The scale of the quantized tensor
    - zero_point: The zero point of the quantized tensor
    - dtype: The data type of the quantized tensor
    """
    scaled_and_shifted_tensor = tensor / scale + zero_point
    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)

    return q_tensor


def linear_dequantization(quantized_tensor, scale, zero_point):
    return scale * (quantized_tensor.float() - zero_point)


def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    """
    A function that calculates the scale and zero point for quantization.

    Args:
    - tensor: The tensor to quantize
    - dtype: The data type of the quantized tensor
    """
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    r_min, r_max = tensor.min().item(), tensor.max().item()

    scale = (r_max - r_min) / (q_max - q_min)

    zero_point = q_min - (r_min / scale)

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        # round and cast to int
        zero_point = int(round(zero_point))

    return scale, zero_point


def get_q_scale_symmetric(tensor, dtype=torch.int8):
    """
    A function that calculates the scale for symmetric quantization.

    Args:
    - tensor: The tensor to quantize
    - dtype: The data type of the quantized tensor
    """
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max

    # return the scale
    return r_max / q_max


def linear_q_symmetric(tensor, dtype=torch.int8):
    """
    A function that quantizes a tensor using symmetric quantization.

    Args:
    - tensor: The tensor to quantize
    - dtype: The data type of the quantized tensor
    """
    scale = get_q_scale_symmetric(tensor)

    quantized_tensor = linear_q_with_scale_and_zero_point(
        tensor,
        scale=scale,
        # in symmetric quantization zero point is = 0
        zero_point=0,
        dtype=dtype,
    )

    return quantized_tensor, scale


def plot_matrix(tensor, ax, title, vmin=0, vmax=1, cmap=None, fmt=".2f"):
    """
    Plot a heatmap of tensors using seaborn

    Args:
    - tensor: The tensor to plot
    - ax: The axis to plot the tensor on
    - title: The title of the plot
    - vmin: The minimum value of the color map
    - vmax: The maximum value of the color map
    - cmap: The color map to use
    - fmt: The format of the annotations
    """
    sns.heatmap(
        tensor.cpu().numpy(),
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        cbar=False,
    )
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def plot_quantization_errors(
    original_tensor, quantized_tensor, dequantized_tensor, dtype=torch.int8, n_bits=8
):
    """
    A method that plots 4 matrices, the original tensor, the quantized tensor
    the de-quantized tensor and the error tensor.

    Args:
    - original_tensor: The original tensor
    - quantized_tensor: The quantized tensor
    - dequantized_tensor: The dequantized tensor
    - dtype: The data type of the quantized tensor
    - n_bits: The number of bits used for quantization
    """
    # Get a figure of 4 plots
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    # Plot the first matrix
    plot_matrix(
        original_tensor, axes[0], "Original Tensor", cmap=ListedColormap(["lightgray"])
    )

    # Get the quantization range and plot the quantized tensor
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    plot_matrix(
        quantized_tensor,
        axes[1],
        f"{n_bits}-bit Linear Quantized Tensor",
        cmap=ListedColormap(["gray"]),
        fmt="d",
    )

    # Plot the de-quantized tensors
    plot_matrix(
        dequantized_tensor,
        axes[2],
        "Dequantized Tensor",
        cmap=ListedColormap(["lightgray"]),
    )

    # Get the quantization errors
    q_error_tensor = abs(original_tensor - dequantized_tensor)
    plot_matrix(
        q_error_tensor,
        axes[3],
        "Quantization Error Tensor",
        vmin=torch.min(q_error_tensor),
        vmax=torch.max(q_error_tensor),
        cmap="coolwarm",
        fmt=".3f",
    )

    fig.tight_layout()
    plt.show()


def linear_q_symmetric_per_channel(r_tensor, dim, dtype=torch.int8):
    """
    A function that quantizes a tensor per channel.

    Args:
    - r_tensor: The tensor to quantize
    - dim: The dimension to quantize along
    - dtype: The data type of the quantized tensor
    """
    output_dim = r_tensor.shape[dim]
    # store the scales
    scale = torch.zeros(output_dim)

    for index in range(output_dim):
        sub_tensor = r_tensor.select(dim, index)
        scale[index] = get_q_scale_symmetric(sub_tensor, dtype=dtype)

    # reshape the scale
    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_q_with_scale_and_zero_point(
        r_tensor, scale=scale, zero_point=0, dtype=dtype
    )

    return quantized_tensor, scale


def linear_q_symmetric_per_group(tensor, group_size, dtype=torch.int8):
    """
    A function that quantizes a tensor per group.

    Args:
    - tensor: The tensor to quantize
    - group_size: The size of the group
    - dtype: The data type of the quantized tensor
    """
    t_shape = tensor.shape
    assert t_shape[1] % group_size == 0
    assert tensor.dim() == 2

    tensor = tensor.view(-1, group_size)

    quantized_tensor, scale = linear_q_symmetric_per_channel(tensor, dim=0, dtype=dtype)

    quantized_tensor = quantized_tensor.view(t_shape)

    return quantized_tensor, scale


def linear_dequantization_per_group(quantized_tensor, scale, group_size):
    """
    A function that dequantizes a quantized tensor per group.

    Args:
    - quantized_tensor: The quantized tensor
    - scale: The scale of the quantized tensor
    - group_size: The size of the group
    """
    q_shape = quantized_tensor.shape
    quantized_tensor = quantized_tensor.view(-1, group_size)

    dequantized_tensor = linear_dequantization(quantized_tensor, scale, 0)

    dequantized_tensor = dequantized_tensor.view(q_shape)

    return dequantized_tensor


def quantization_error(original_tensor, dequantized_tensor):
    """
    A function that calculates the quantization error between the original tensor and the dequantized tensor.

    Args:
    - original_tensor: The original tensor
    - dequantized_tensor: The dequantized tensor
    """
    return (original_tensor - dequantized_tensor).square().mean()


def w8_a16_forward(weight, input, scales, bias=None):
    """
    A function that performs the forward pass of a linear layer with 8-bit weights and 16-bit scales.

    Args:
    - weight: The quantized weights of the linear layer
    - input: The input tensor
    - scales: The scales of the linear layer
    - bias: The bias of the linear layer
    """
    casted_weights = weight.to(input.dtype)
    output = F.linear(input, casted_weights) * scales

    if bias is not None:
        output = output + bias

    return output


class W8A16LinearLayer(nn.Module):
    """
    A class that implements a linear layer with 8-bit weights and 16-bit scales.

    The class initializes the weights and scales randomly and quantizes the weights
    during initialization.

    The forward pass of the layer uses the quantized weights and scales to compute the output.
    """

    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        super().__init__()

        self.register_buffer(
            "int8_weights",
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8),
        )

        self.register_buffer("scales", torch.randn((out_features), dtype=dtype))

        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))

        else:
            self.bias = None

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights / scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales

    def forward(self, input):
        return w8_a16_forward(self.int8_weights, input, self.scales, self.bias)


def replace_linear_with_target_and_quantize(
    module, target_class, module_name_to_exclude
):
    """
    A function that replaces all the linear layers in a module with a target class
    and quantizes the weights of the new layer.

    Args:
    - module: The module to traverse
    - target_class: The target class to replace the linear layers with
    - module_name_to_exclude: A list of module names to exclude from the replacement
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not any(
            [x == name for x in module_name_to_exclude]
        ):
            old_bias = child.bias
            old_weight = child.weight

            new_module = target_class(
                child.in_features,
                child.out_features,
                old_bias is not None,
                child.weight.dtype,
            )
            setattr(module, name, new_module)

            getattr(module, name).quantize(old_weight)

            if old_bias is not None:
                getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(
                child, target_class, module_name_to_exclude
            )


# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def plot_results(model, pil_img, results):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    scores, labels, boxes = results["scores"], results["labels"], results["boxes"]
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(
        scores.tolist(), labels.tolist(), boxes.tolist(), colors
    ):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        text = f"{model.config.id2label[label]}: {score:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()
