# gradcam.py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initializes the GradCAM module.

        Args:
            model (nn.Module): Your model.
            target_layer (nn.Module): The layer from which to extract activations and gradients.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        # Forward hook: save activations.
        def forward_hook(module, input, output):
            # If the output is a tuple, assume the first element is the tensor of interest.
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        # Backward hook: save gradients.
        def backward_hook(module, grad_in, grad_out):
            # Similarly, if grad_out is a tuple, take the first element.
            if isinstance(grad_out, tuple):
                self.gradients = grad_out[0].detach()
            else:
                self.gradients = grad_out.detach()
        
        # Register the hooks on the target layer.
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def __call__(self, input_tensor, target_scalar=None):
        """
        Generates the GradCAM map for the given input.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor of shape [B, C, H, W].
            target_scalar (torch.Tensor, optional): A scalar for the target; if None, the sum over outputs is used.
        
        Returns:
            grad_cam_map (torch.Tensor): The normalized heatmap upsampled to the input image size.
        """
        self.model.zero_grad()
        # Perform a forward pass.

        out = self.model(input_tensor)

        # We are only interested in the first element
        if isinstance(out, tuple) and len(out) == 3:
            output, _, _ = out
        else:
            output = out
        
        # Define a scalar target. (Customize this if you want to focus on a specific output.)
        if target_scalar is None:
            target = output.sum()
        else:
            target = target_scalar
        
        target.backward(retain_graph=True)
        
        # Compute the importance weights: global average pooling over gradients.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # Compute weighted combination of the activations.
        grad_cam_map = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        
        # Upsample the GradCAM map to the size of the input image.
        grad_cam_map = F.interpolate(grad_cam_map, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize the map for visualization.
        grad_cam_map = grad_cam_map - grad_cam_map.min()
        if grad_cam_map.max() != 0:
            grad_cam_map = grad_cam_map / grad_cam_map.max()
        
        return grad_cam_map

def show_heatmap_on_image(image, heatmap, title):
    """
    Overlays a heatmap on an image and displays it.
    
    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W].
        heatmap (torch.Tensor): Heatmap tensor of shape [1, H, W].
        title (str): Title for the plot.
    """
    image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
    heatmap_np = heatmap.detach().cpu().numpy().squeeze()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np, cmap='gray')
    plt.imshow(heatmap_np, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.show()
