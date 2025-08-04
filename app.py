import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # Reduce size for speed
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# Postprocessing
def postprocess(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image / 255
    image = torch.clamp(image, 0, 1)
    return transforms.ToPILImage()(image)

# Load image
def load_image(img):
    image = preprocess(img).unsqueeze(0).to(device, torch.float)
    return image

# Gram matrix calculation
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

# Feature extractor
class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:22]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name == "0": features["conv1_1"] = x
            if name == "5": features["conv2_1"] = x
            if name == "10": features["conv3_1"] = x
            if name == "19": features["conv4_1"] = x
            if name == "21": features["conv4_2"] = x
        return features

# Style transfer function
def style_transfer(content_img, style_img, steps=150, style_weight=1e6, content_weight=1):
    content = load_image(content_img)
    style = load_image(style_img)
    generated = content.clone().requires_grad_(True)

    model = VGGFeatures().to(device)
    optimizer = torch.optim.LBFGS([generated])

    content_features = model(content)
    style_features = model(style)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features if layer != 'conv4_2'}

    run = [0]
    while run[0] <= steps:
        def closure():
            optimizer.zero_grad()
            gen_features = model(generated)

            content_loss = F.mse_loss(gen_features["conv4_2"], content_features["conv4_2"])

            style_loss = 0
            for layer in style_grams:
                gen_gram = gram_matrix(gen_features[layer])
                style_loss += F.mse_loss(gen_gram, style_grams[layer])

            total_loss = content_weight * content_loss + style_weight * style_loss
            total_loss.backward()
            run[0] += 1
            return total_loss
        optimizer.step(closure)

    return postprocess(generated)

# Gradio app
app = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(type="pil", label="Upload Content Image"),
        gr.Image(type="pil", label="Upload Style Image")
    ],
    outputs=gr.Image(label="Stylized Output"),
    title="Neural Style Transfer",
    description="Upload content & style images to generate an artistic blend."
)

app.launch()
