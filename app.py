import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing and Postprocessing
loader = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

unloader = transforms.Compose([
    transforms.Lambda(lambda x: x.div(255)),
    transforms.ToPILImage()
])

def load_image(img):
    image = loader(img).unsqueeze(0)
    return image.to(device, torch.float)

def gram_matrix(input):
    batch_size, channel, height, width = input.size()
    features = input.view(channel, height * width)
    G = torch.mm(features, features.t())
    return G.div(channel * height * width)

# function for  VGG feature extractor
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = nn.Sequential(*list(vgg.children())[:21]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False

        self.selected_layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2'}

    def forward(self, x):
        features = {}
        for name, layer in self.layers._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features[self.selected_layers[name]] = x
        return features

# Styling transfer logic
def style_transfer(content_img, style_img, num_steps=50, style_weight=1e6, content_weight=1):
    content = load_image(content_img)
    style = load_image(style_img)
    generated = content.clone().requires_grad_(True)

    model = VGGFeatureExtractor().to(device)
    optimizer = torch.optim.LBFGS([generated])

    content_features = model(content)
    style_features = model(style)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    run = [0]
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            gen_features = model(generated)

            content_loss = F.mse_loss(gen_features['conv4_2'], content_features['conv4_2'])
            style_loss = 0
            for layer in style_grams:
                gen_feat = gen_features[layer]
                gen_gram = gram_matrix(gen_feat)
                style_gram = style_grams[layer]
                style_loss += F.mse_loss(gen_gram, style_gram)

            total_loss = style_weight * style_loss + content_weight * content_loss
            total_loss.backward()
            run[0] += 1
            return total_loss
        optimizer.step(closure)

    output = generated.squeeze().cpu().clone().detach()
    output = unloader(output)
    return output

# Gradio interface to deploy in hugging face spaces
app = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(type="pil",label="Upload Content Image"),
        gr.Image(type="pil",label="Upload Style Image")
    ],
    outputs=gr.Image(label="Stylized Output"),
    title="Neural Style Transfer",
    description="Upload a content and style image to generate stunning, stylized output in seconds"
)

app.launch()
