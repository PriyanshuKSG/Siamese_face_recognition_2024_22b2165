import torchvision
import torchinfo 

import torchvision, torchinfo
def get_model_from_pytorch(model_weights,
                           model_fn: str,
                           device: str,
                           image_size: int,
                          show_summary: bool = True):
    """
    Loads the pre-trained model from Pytorch's torchvision.models and returns the required model and its transforms.
    Please see Example usuage at the end of the docstring.
    
    Input Arguments:
    1) model_weights: Model weights (.DEFAULT) needed to be downloaded.
    Refer: https://pytorch.org/vision/main/models.html#classification
    2) model_fn: Model function name as string. Refer to example usuage.
    3) image_size: Integer denoting the height of the image. By default, height = width for the image

    4) show_summary: Uses "summary" from torchinfo to show the outline of the model. Default = True.

    Output/Returns:
     - model = pre-trained model
     - train_transform = transform used by the model when it was originally trained
     - test_transform = transform to be applied on validation and test images

    Example usuage:
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
    model_fn_string = "torchvision.models.efficientnet_b3"
    model, train_transform, test_transform = get_model_from_pytorch(model_weights=weights,
                                                                    model_fn=model_fn_string,
                                                                    device=device,
                                                                    image_size=300,
                                                                    show_summary=True)
    """
    load_model_fn = eval(model_fn)
    model = load_model_fn(weights=model_weights)
    train_transform = model_weights.transforms()
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=image_size),
        torchvision.transforms.CenterCrop(size=image_size),
        torchvision.transforms.ToTensor()
    ])
    for param in model.parameters():
        param.requires_grad = False

    if show_summary:
        print(torchinfo.summary(model, input_size=(32, 3, image_size, image_size),
                         col_names = ['input_size', 'output_size', 'num_params', 'trainable']))
    return model, train_transform, test_transform