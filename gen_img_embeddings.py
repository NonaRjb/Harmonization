import tensorflow as tf
from PIL import Image
import numpy as np
import os
import argparse
from harmonization.models import (load_ViT_B16, load_ViT_B16_orig, 
                                  load_ResNet50,
                                  load_VGG16, load_EfficientNetB0,
                                  load_tiny_ConvNeXT, load_tiny_MaxViT,
                                  load_LeViT_small,
                                  preprocess_input)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Image Embeddings")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--save_path", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--model_type", type=str, default="clip_vitb32", help="Type of model to use")
    parser.add_argument("--model_name", type=str, default="dreamsim", help="Name of the method")
    return parser.parse_args()

def load_img_encoder(model_type):
    if model_type == "vitb16":
        return load_ViT_B16(include_top=False)
    elif model_type == "vitb16_noalign":
        return load_ViT_B16_orig(include_top=False)
    else:
        raise ValueError(f"Model type {model_type} not recognized")

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

# vit_harmonized = load_ViT_B16(include_top=False)
# vit_orig = load_ViT_B16_orig(include_top=False)
# vgg_harmonized = load_VGG16()
# resnet_harmonized = load_ResNet50()
# efficient_harmonized = load_EfficientNetB0()
# convnext_harmonized = load_tiny_ConvNeXT()
# maxvit_harmonized = load_tiny_MaxViT()
# levit_harmonized = load_LeViT_small()

# load images (in [0, 255])
# ...
# img_file = "/proj/rep-learning-robotics/users/x_nonra/eeg_asif_img/data/things_eeg_2/images/test_images/00001_aircraft_carrier/aircraft_carrier_06s.jpg"
# pil_image = Image.open(img_file)
# rgb_image = _convert_image_to_rgb(pil_image)
# # Convert PIL image to TensorFlow tensor
# image_tensor = tf.keras.utils.img_to_array(rgb_image)
# # Apply transformations
# transform = _transform(224)
# processed_image = transform(image_tensor)
# processed_image = tf.expand_dims(processed_image, axis=0)
# processed_image = preprocess_input(np.asarray(processed_image))
# predictions_harmonized = vit_harmonized(processed_image)
# predictions_orig = vit_orig(processed_image)

args = parse_args()

transform = _transform(224)
model = load_img_encoder(args.model_type)
data_path = args.data_path
save_path = data_path if args.save_path is None else args.save_path
img_parent_dir  = os.path.join(data_path, 'images')
img_parent_save_dir = os.path.join(save_path)
img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()
train_embeddings = []
for item in range(16540):
    img_file = os.path.join(img_parent_dir, 'training_images', 
                    img_metadata['train_img_concepts'][item], img_metadata['train_img_files'][item])
    img = Image.open(img_file).convert('RGB')
    img = tf.keras.utils.img_to_array(img)
    img = transform(img)
    img = tf.expand_dims(img, axis=0)
    img = preprocess_input(np.asarray(img))
    e = tf.stop_gradient(model(img).numpy())
    train_embeddings.append(e)
    # np.save(img_file.replace(".jpg", f"_{args.model_name}_{args.model_type}.npy"), e)
    if item % 1000 == 0:
        print(f"{item} items out of 16540 done")
        print(f"e.shape = {e.shape}")
train_embeddings = np.array(train_embeddings).squeeze()
print(f"train_embeddings.shape = {train_embeddings.shape}")
np.save(os.path.join(img_parent_save_dir, f"train_{args.model_name}_{args.model_type}.npy"), train_embeddings)
print("Start Embedidng Test Images")
test_embeddings = []
for item in range(200):
    img_file = os.path.join(img_parent_dir, 'test_images', 
                    img_metadata['test_img_concepts'][item], img_metadata['test_img_files'][item])
    img = Image.open(img_file).convert('RGB')
    img = tf.keras.utils.img_to_array(img)
    img = transform(img)
    img = tf.expand_dims(img, axis=0)
    img = preprocess_input(np.asarray(img))
    e = tf.stop_gradient(model(img).numpy())
    test_embeddings.append(e)
    # np.save(img_file.replace(".jpg", f"_{args.model_name}_{args.model_type}.npy"), e)
test_embeddings = np.array(test_embeddings).squeeze()
print(f"test_embeddings.shape = {test_embeddings.shape}")
np.save(os.path.join(img_parent_save_dir, f"test_{args.model_name}_{args.model_type}.npy"), test_embeddings)