# import required packages

import os
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from PIL import Image
import io
from torch.utils.data import Dataset
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
import pickle
import glob
import sys


# Define MediaValet Dataset Class
class MediaValetDataset(Dataset):

    def __init__(self, image_bytes, transform):
        self.transform = transform
        self.files = [image_bytes]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_bytes_array = self.files[idx]
        image = Image.open(io.BytesIO(image_bytes_array)).convert("RGB")

        return self.transform(image)


# Define Macros
num_matches = 25
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load resnet pretrained model
model = models.resnet50(pretrained=True)
model.eval()
model = model.to(DEVICE)
cut_model = nn.Sequential(*list(model.children())[:-1])


# Define Get Matches Function
def get_matches(pair_dists, selected_indicies, num_matches=10):
    k = min(num_matches, selected_indicies.shape[0])
    dists, inds = torch.topk(pair_dists[selected_indicies, :], k, dim=0, sorted=True)
    return (selected_indicies[inds].detach().cpu().numpy(), dists.detach().cpu().numpy())


# Get Final Result in json format
def get_result(test_image_bytes, reference_images_pickle, diff_pickle):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Use Test image path
    dataset = MediaValetDataset(test_image_bytes, data_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Generate features and tensor for test image
    for i, (inputs) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(DEVICE)
        test_image_feature = torch.squeeze(cut_model(inputs)).detach().cpu().numpy()
        test_image_feature = np.reshape(test_image_feature, (1, test_image_feature.size))

    test_image_tensor = torch.from_numpy(test_image_feature).float().to("cpu:0")
    test_image_tensor = test_image_tensor / torch.sqrt(torch.sum(test_image_tensor ** 2, dim=1, keepdim=True))
    test_image_tensor = test_image_tensor.to(DEVICE)
    
    if os.path.isfile(diff_pickle):
        return load_and_match((test_image_tensor, diff_pickle)) + load_and_match((test_image_tensor, reference_images_pickle)) 
    else:
        return load_and_match((test_image_tensor, reference_images_pickle))

def load_and_match(args):
    (test_tensor, pkl_file) = args
    with open(pkl_file, "rb") as f:
        (reference_images_features, reference_images_attributes) = pickle.load(f)

    
    reference_images_tensor = torch.from_numpy(reference_images_features).float().to("cpu:0")
    reference_images_tensor = reference_images_tensor / torch.sqrt(
        torch.sum(reference_images_tensor ** 2, dim=1, keepdim=True))
    reference_images_tensor = reference_images_tensor.to(DEVICE)

    indicies = torch.arange(0, reference_images_tensor.shape[0]).to(DEVICE)

    # Get top-k matching images in JSON format
    pair_dists = torch.einsum("nf,bf->nb", reference_images_tensor, test_tensor)
    matched_indicies, matched_scores = get_matches(pair_dists, indicies, num_matches=num_matches)

    dict_match = []
    for i in range(len(matched_indicies)):
        filename = reference_images_attributes["orig_name"][matched_indicies[i][0]].split("/")[-1]
        sim_score = round(matched_scores[i][0], 2)
        
        # condition for duplicate images       
        if sim_score>=0.99:        
            image_type="duplicate"
        # condition for derivative images
        elif sim_score >= 0.93:
            image_type="derivative"     
        else:
            image_type=""
            
        if sim_score<0.85:
            pass 
        else:
            dict_ind = {"filename": str(filename), "score": str(sim_score), "type": str(image_type)}
            dict_match.append(dict_ind)
            
    return dict_match

