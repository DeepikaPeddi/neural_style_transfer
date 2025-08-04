import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. Image Loading & Preprocessing
def load_image(path, max_size=512):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),  # Ensure 3 channels
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dim
    return image.to(device)

#2. Gram Matrix Calculation
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)

# 3. VGG Feature Extractor
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.selected_layers = {'0': 'conv_1', '5': 'conv_2', '10': 'conv_3', '19': 'conv_4', '28': 'conv_5'}
        self.model = nn.Sequential(*[vgg[i] for i in range(29)])
        
    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                features[self.selected_layers[name]] = x
        return features

# 4. Load images
content_path = "content.jpg"
style_path = "style.jpg"
content_image = load_image(content_path)
style_image = load_image(style_path)

# 5. Extract Features
vgg = VGGFeatureExtractor().to(device).eval()
content_features = vgg(content_image)
style_features = vgg(style_image)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

#  6. Initialize Generated Image 
generated = content_image.clone().requires_grad_(True)

#  7. Define Loss & Optimizer 
style_weight = 1e6
content_weight = 1e0
optimizer = optim.LBFGS([generated])
num_steps = 300

# 8. Optimization Loop 
print("Starting Style Transfer...")
run = [0]
while run[0] <= num_steps:
    def closure():
        optimizer.zero_grad()
        gen_features = vgg(generated)
        
        # Content loss
        content_loss = F.mse_loss(gen_features["conv_4"], content_features["conv_4"])
        
        # Style loss
        style_loss = 0
        for layer in style_grams:
            gen_gram = gram_matrix(gen_features[layer])
            style_gram = style_grams[layer]
            style_loss += F.mse_loss(gen_gram, style_gram)
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()

        if run[0] % 50 == 0:
            print(f"Step {run[0]}: Total Loss: {total_loss.item():.4f}")
        
        run[0] += 1
        return total_loss

    optimizer.step(closure)

print("Style Transfer Completed!")

# 9. Converting Tensor to Image
def tensor_to_image(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.Normalize(
        mean=[-2.118, -2.036, -1.804],  # reverse normalization
        std=[4.367, 4.464, 4.444]
    )(image)
    image = torch.clamp(image, 0, 1)
    return transforms.ToPILImage()(image)

# 10. displaying result
result = tensor_to_image(generated)
plt.imshow(result)
plt.axis('off')
plt.title("Stylized Output")
plt.show()

result.save("stylized_output.jpg")
print("Stylized image saved as 'stylized_output.jpg'")
