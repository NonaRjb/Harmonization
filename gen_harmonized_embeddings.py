import tensorflow as tf
from tensorflow.keras.models import Model
import torch
from torchsummary import summary
import timm
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from timm.data import resolve_data_config
from PIL import Image
import numpy as np
import os
import argparse
from harmonization.models import (load_ViT_B16,
                                  load_ResNet50,
                                  load_VGG16, load_EfficientNetB0,
                                  load_tiny_ConvNeXT, load_tiny_MaxViT,
                                  load_LeViT_small,
                                  preprocess_input)

model_configs = {
    'resnet50': {
        'penultimate_layer': 'avg_pool',
        'return_node': 'global_pool.flatten'
    },
    'vitb16': {
        'return_node': 'fc_norm'
    },
    'vgg16': {
        'penultimate_layer': 'fc2',
    },
    'efficientnetb0': {
        'penultimate_layer': 'avg_pool'
    },
    'convnext': {
        'penultimate_layer': 'head_ln',
        'return_node': 'head.flatten'
    },
    'maxvit': {
        'penultimate_layer': 'features_tanh'
    },
    'levit': {
        'penultimate_layer': 'flatten',
        'return_node': 'mean'
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_dir",
        default="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_img_embeddings/harmonization"
    )
    parser.add_argument(
        "--data_root",
        default="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/images"
    )
    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--human_aligned", action="store_true")

    return parser.parse_args()


def _transform(n_px):
    def preprocess(image):
        # Resize image
        image = tf.image.resize(image, [n_px, n_px], method='bicubic')
        
        # Center crop: if image is not square, crop to center square
        if image.shape[0] != image.shape[1]:
            crop_size = min(image.shape[0], image.shape[1])
            offset_height = (image.shape[0] - crop_size) // 2
            offset_width = (image.shape[1] - crop_size) // 2
            image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_size, crop_size)
        
        return image
    
    return preprocess


class OriginalImageEncoder(torch.nn.Module):
    def __init__(self, model_type):
        super(OriginalImageEncoder, self).__init__()
        self.model, _ = load_original_img_encoder(model_type)
        print(get_graph_node_names(self.model))
        self.return_node = model_configs[model_type]['return_node']
        self.model.eval()
        self.feature_extractor = create_feature_extractor(self.model, return_nodes=[self.return_node])
        # print(f"model = {self.model}")

    def forward(self, x):
        x = self.feature_extractor(x)[self.return_node]
        # x = self.model(x)
        return x

def load_original_img_encoder(model_type):
    if model_type == "vitb16":
        encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
    elif model_type == "resnet50":
        encoder = timm.create_model('resnet50', pretrained=True)
    elif model_type == "vgg16":
        encoder = timm.create_model('vgg16', pretrained=True)
    elif model_type == "efficientnetb0":
        encoder = timm.create_model('tf_efficientnet_b0', pretrained=True)
    elif model_type == "convnext":
        encoder = timm.create_model('convnext_tiny', pretrained=True)
    elif model_type == "maxvit":
        encoder = timm.create_model('maxvit_tiny_tf_224', pretrained=True)
    elif model_type == "levit":
        encoder = timm.create_model('levit_conv_128', pretrained=True)

    config = resolve_data_config({}, model=encoder)
    transform = create_transform(**config, is_training=False)
    print(f"transform = {transform}")
    preprocess = transforms.Compose([transform])
    return encoder, preprocess

def load_harmonized_img_encoder(model_type):
    if model_type == "vitb16":
        return load_ViT_B16(include_top=False)
    elif model_type == "resnet50":
        return load_ResNet50()
    elif model_type == "vgg16":
        return load_VGG16()
    elif model_type == "efficientnetb0":
        return load_EfficientNetB0()
    elif model_type == "convnext":
        return load_tiny_ConvNeXT()
    elif model_type == "maxvit":
        return load_tiny_MaxViT()
    elif model_type == "levit":
        return load_LeViT_small()
    else:
        raise ValueError(f"Model type {model_type} not recognized")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.embeddings_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = _transform(224)

    if args.human_aligned:
        model = load_harmonized_img_encoder(args.model_name)
        print(model.summary())
        if 'penultimate_layer' in model_configs[args.model_name].keys():
            penultimate_layer = Model(inputs=model.input, outputs=model.get_layer(model_configs[args.model_name]['penultimate_layer']).output)
    else:
        model = OriginalImageEncoder(args.model_name)
        _, preprocess = load_original_img_encoder(args.model_name)

    data_path = args.data_root
    save_path = data_path if args.embeddings_dir is None else args.embeddings_dir
    img_parent_dir  = data_path
    img_parent_save_dir = save_path
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()

    train_embeddings = []
    for item in range(16540):
        img_file = os.path.join(img_parent_dir, 'training_images', 
                        img_metadata['train_img_concepts'][item], img_metadata['train_img_files'][item])
        img_save_file = os.path.join(img_parent_save_dir, 'training_images', 
                        img_metadata['train_img_concepts'][item], img_metadata['train_img_files'][item])
        img = Image.open(img_file).convert('RGB')
        if args.human_aligned:
            img = transform(img)
            img = tf.expand_dims(img, axis=0)
            img = preprocess_input(img)
            if 'penultimate_layer' in model_configs[args.model_name].keys():
                e = tf.stop_gradient(penultimate_layer.predict(img))
            else:
                e = tf.stop_gradient(model(img).numpy())
        else:
            img = preprocess(img)
            img = img.unsqueeze(0)
            with torch.no_grad():
                e = model(img).detach().cpu().numpy()

        # train_embeddings.append(e)
        if args.human_aligned:
            np.save(img_save_file.replace(".jpg", f"_harmonization_{args.model_name}.npy"), e)
        else:
            np.save(img_save_file.replace(".jpg", f"_harmonization_{args.model_name}_noalign.npy"), e)
        if item % 1000 == 0:
            print(f"{item} items out of 16540 done")
            print(f"e.shape = {e.shape}")

    print("Start Embedidng Test Images")
    test_embeddings = []
    for item in range(200):
        img_file = os.path.join(img_parent_dir, 'test_images', 
                        img_metadata['test_img_concepts'][item], img_metadata['test_img_files'][item])
        img_save_file = os.path.join(img_parent_save_dir, 'test_images', 
                        img_metadata['test_img_concepts'][item], img_metadata['test_img_files'][item])
        img = Image.open(img_file).convert('RGB')
        if args.human_aligned:
            img = transform(img)
            img = tf.expand_dims(img, axis=0)
            img = preprocess_input(img)
            if 'penultimate_layer' in model_configs[args.model_name].keys():
                e = tf.stop_gradient(penultimate_layer.predict(img))
            else:
                e = tf.stop_gradient(model(img).numpy())
        else:
            img = preprocess(img)
            img = img.unsqueeze(0)
            with torch.no_grad():
                e = model(img).detach().cpu().numpy()

        # train_embeddings.append(e)
        if args.human_aligned:
            np.save(img_save_file.replace(".jpg", f"_harmonization_{args.model_name}.npy"), e)
        else:
            np.save(img_save_file.replace(".jpg", f"_harmonization_{args.model_name}_noalign.npy"), e)
    
    print("Done!")
    # extractor = get_extractor(
    #     model_name='Harmonization',
    #     source='custom',
    #     device=device,
    #     pretrained=True,
    #     model_parameters={'variant': args.model_name},
    #     )

    # batch_size = 32

    # dataset = ImageDataset(
    #     root=data_root,
    #     out_path=args.embeddings_dir,
    #     backend=extractor.get_backend(), # backend framework of model
    #     transforms=extractor.get_transformations(resize_dim=224, crop_dim=224) # set the input dimensionality to whichever values are required for your pretrained model
    # )

    # batches = DataLoader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     backend=extractor.get_backend() # backend framework of model
    # )

    # features = extractor.extract_features(
    #     batches=batches,
    #     module_name='Transformer/encoderblock_11/Dense_1',
    #     flatten_acts=False,
    #     output_type="ndarray", # or "tensor" (only applicable to PyTorch models of which CLIP and DINO are ones!)
    # )

    # # if "dino" in args.model_name.lower():
    # #     features = features[:, -1, :].squeeze()
    # print(features.shape)

    # # save_features(features, out_path=args.embeddings_dir, file_format='npy') # file_format can be set to "npy", "txt", "mat", "pt", or "hdf5"

    # # os.rename(os.path.join(args.embeddings_dir, f"features.npy"), os.path.join(args.embeddings_dir, f"{split}_gLocal_{args.model_name.lower()}_noalign.npy"))
    # # os.rename(os.path.join(args.embeddings_dir, f"file_names.txt"), os.path.join(args.embeddings_dir, f"filenames_{split}_gLocal_{args.model_name.lower()}_noalign.txt"))