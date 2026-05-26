import torch
import timm
from timm.data import resolve_data_config, create_transform
import numpy as np
from evals.spin_modelling import compute_pseudolikelihood, convert_labels_to_spins
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def load_multilabel_classifier(classifier_path, device='cuda'):
    print(f"Loading checkpoint from: {classifier_path}")
    checkpoint = torch.load(classifier_path, map_location=device)
    
    # Rebuild Model Architecture
    model_name = checkpoint.get("initialize_timm_model", "convnext_nano")
    num_classes = checkpoint.get("num_classes", 40)
    
    model = timm.create_model(model_name, pretrained=False)
    model.reset_classifier(num_classes=num_classes)
    
    # Load Weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model
  

def predict_attributes(model, image_tensor, device='cuda'):
    """
    Predicts attributes for an image or batch of images.
    """
    # create transform to mimic resizing+normalization used during finetuning
    transform = T.Compose([
        T.Resize(73, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop((64, 64)),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # Add batch dim if single image
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = transform(image_tensor)

        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
    return probs, preds


def compute_loglikelihood_samples(samples, batch_size=128, ising_params_path='evals/ising_params.npz', 
                                  classifier_path='evals/checkpoint_epoch_0006.pt', ind_excluded=[21, 37], device='cuda'):
    '''
    Estimates loglikelihood of a batch of samples using the multilabel classifier and Ising model
    samples: torch tensor, contains images
    batch_size: int, batch size to pass through classifier
    ising_params_path: str, path to file containing h, J
    classifier_path: str, path to classifier checkpoint
    ind_excluded: list of int, indices of attributes to exclude from log-likelihood computation (21=mouth_slightly_open, 37=wearing_necklace)
    '''
    model = load_multilabel_classifier(classifier_path)
    # indices of attributes to keep
    ind_keep = torch.tensor([i for i in range(40) if i not in ind_excluded])

    # load Ising parameters
    ising_params = np.load(ising_params_path)
    h = ising_params['h']
    J = ising_params['J']
  
    all_preds = []

    # get  classifier predictions
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size].to(device)
        probs, preds = predict_attributes(model, batch)
        all_preds.append(preds.cpu())

    # combine all predictions
    preds = torch.cat(all_preds, dim=0)

    # compute log-liekilhood
    pl_est = compute_pseudolikelihood(
        convert_labels_to_spins(preds.cpu().numpy()[:, ind_keep]), h, J
    )

    print(f'Average log-likelihood: {np.mean(pl_est)}')

    return pl_est




