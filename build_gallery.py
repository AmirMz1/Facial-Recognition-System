import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random
import cv2
# import tensorflow as tf
import pickle
from collections import defaultdict


def load_and_preprocess_images(dataset_dir):
    """
    Loads images from the dataset directory, detects and crops faces,
    resizes them, and optionally applies occlusion.

    Parameters:
    - dataset_dir: Path to the dataset directory
    - apply_occlusion_flag: Whether to apply occlusion to the images

    """
    counter = 0
    for root, _, filenames in os.walk(dataset_dir):
        for img_filename in filenames:
            # if img_filename.endswith('.pgm') and not img_filename.replace('.pgm', '.npy') in filenames:
            if img_filename.endswith('.jpg') and not img_filename.replace('.jpg', '.npy') in filenames:


                img_path = os.path.join(root, img_filename)
                if not os.path.exists(img_path.replace('jpg', 'npy')):
                    img = cv2.imread(img_path)


                    if img is None and np.max(img) != 0:
                        print(f"Failed to load image: {img_path}")
                        continue


                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Resize the image to a standard size
                    resized_img = cv2.resize(img, (128, 128))

                    np.save(img_path.replace('jpg', 'npy'), np.array(resized_img))
                    print(f"{img_filename.replace('jpg', 'npy')} saved!")


                    counter+=1


def mirror_image(image):
    """Apply horizontal mirroring to the image."""
    return np.flip(image, axis=1)

def add_blur(image, kernel_size=5):
    """Apply a small Gaussian blur to the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 1)

def add_noise(image, mean=0, std=0.05):
    """Add small Gaussian noise to the image."""
    image = image / 255.0  # Normalize to [0, 1]
    noise = np.random.normal(mean, std, image.shape)
    noisy_img = image + noise
    noisy_img = np.clip(noisy_img, 0, 1)
    return (noisy_img * 255).astype(np.uint8)  # Return to [0, 255]

def adjust_contrast(image, alpha=1.3, beta=0):
    """Adjust contrast of the image."""
    image = image / 255.0  # Normalize
    contrasted_img = alpha * image + beta
    contrasted_img = np.clip(contrasted_img, 0, 1)
    return (contrasted_img * 255).astype(np.uint8)

def load_and_preprocess_npy(dataset_dir, outNamePrefix, unique_labels=None):
    features = []
    labels = []
    # skip_folders = {'recon', 'steps', 'latent'}
    labels_ = unique_labels


    for root, _, filenames in os.walk(dataset_dir):
        # Skip folders if the last part of the path is in skip_folders
        if os.path.basename(root) in labels_:
            for img_filename in filenames:
                state = random.choice([True, False, False])
                # if img_filename.endswith('.npy') and (not 'eyeOcclu' in img_filename or not 'lipOcclu' in img_filename):                              # *******************
                if img_filename.endswith('.npy'):                              # *******************

                    class_label = os.path.basename(root)


                    img_path = os.path.join(root, img_filename)
                    img = np.load(img_path)


                    # Original image
                    features.append(img)
                    labels.append(class_label)
                    if outNamePrefix == 'train':
                        # Mirrored image
                        mirrored_img = mirror_image(img)
                        features.append(mirrored_img)
                        labels.append(class_label)

                        if state:
                            # Blurred image
                            blurred_img = add_blur(img)
                            features.append(blurred_img)
                            labels.append(class_label)

                            # Noisy image
                            noisy_img = add_noise(img)
                            features.append(noisy_img)
                            labels.append(class_label)

                            # Contrast-adjusted image
                            contrast_img = adjust_contrast(img)
                            features.append(contrast_img)
                            labels.append(class_label)

    # Convert to NumPy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Map string labels to integers
    unique_labels = np.unique(labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    labels = np.array([label_to_int[label] for label in labels], dtype=np.float32)

    # # Save processed data
    if not os.path.exists('data/'):
        os.mkdir('data/')

    np.save(f'data/{outNamePrefix}_data.npy', features)
    np.save(f'data/{outNamePrefix}_labels.npy', labels)

    print("\nDatasets saved as .npy formats with augmentations applied.")

    # Return label mapping dictionary
    return label_to_int



# if '__name__' == '__main__':
if not os.path.exists("data/train_labels.npy"):
    dataset_dir = 'CelebA_HQ_facial_aligment/train'
    # preprocessing pgm files and convert to .npy format each file
    # load_and_preprocess_images(dataset_dir)
    names = load_and_preprocess_npy(dataset_dir, outNamePrefix='train', unique_labels=os.listdir(dataset_dir))
    print("labels:", names)

if not os.path.exists("data/test_labels.npy"):
    dataset_dir = 'CelebA_HQ_facial_aligment/test'
    # preprocessing pgm files and convert to .npy format each file
    # load_and_preprocess_images(dataset_dir)
    names = load_and_preprocess_npy(dataset_dir, outNamePrefix='test', unique_labels=os.listdir('CelebA_HQ_facial_aligment/train'))
    print("labels:", names)


import numpy as np
from sklearn.utils import shuffle

output_data = "data"
# Load the dataset
features = np.load(f'{output_data}/train_data.npy')
labels = np.load(f'{output_data}/train_labels.npy')

print("data samples:", len(labels))


# Check if any images were loaded
if features.size == 0 or labels.size == 0:
    print("No images were loaded. Please check the dataset directory and file paths.")
else:
    print(f"Loaded {features.shape[0]} images with shape {labels.shape[1:]}.")

    # Reshape data for the model
    features = features.reshape(-1, features.shape[1], features.shape[2], 1)
    # Convert grayscale images to 3-channel images
    features = np.concatenate([features] * 3, axis=-1)


    from model import *

    # Step 1: Rebuild the model architecture
    n_classes = 17
    model = load_model(n_classes)
    # Step 2: Load the saved weights
    model.load_weights("checkpoints/hamface_model.h5")
    # (Optional) Verify model is working
    print("Model loaded successfully.")
    # Recreate the HAMFaceLoss instance
    loss_fn = HAMFaceLoss(num_classes=n_classes)
    # Load the saved weight matrix and assign it to loss_fn
    class_weights = np.load("checkpoints/hamface_class_weights.npy")
    loss_fn.W.assign(class_weights)


    # embeddings = model.predict([features, features])
    batch_size = 16
    train_dataset = tf.data.Dataset.from_tensor_slices(((features, features), labels)).batch(batch_size)

    # Define gallery dictionary
    gallery = defaultdict(list)

    # Loop over dataset in batches and collect normalized embeddings
    for (features1, features2), labels in train_dataset:
        features1 = tf.cast(features1, tf.float32)
        features2 = tf.cast(features2, tf.float32)
        labels = tf.cast(labels, tf.float32)
        
        # Ensure inputs are on the right device
        embeddings = model.predict([features1, features2])
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

        for embedding, label in zip(embeddings, labels.numpy().astype(int)):
            gallery[label].append(embedding.numpy())


    # Cast labels to int before the loop
    labels = tf.cast(labels, tf.int32)

    # Loop over embeddings and corresponding labels to populate the gallery
    for embedding, label in zip(embeddings, labels):
        gallery[int(label.numpy())].append(embedding.numpy())


    # Optionally, average the embeddings per class for a compact gallery
    gallery_avg = {label: np.mean(np.array(emb_list), axis=0) for label, emb_list in gallery.items()}

    # Save the full gallery (all embeddings per person)
    with open(f'{output_data}/gallery_full.pkl', 'wb') as f:
        pickle.dump(gallery, f)
        print("Full gallery (with all embeddings) saved to data/gallery_full.pkl")

    # Save the average embedding gallery (one embedding per person)
    with open(f'{output_data}/gallery_avg.pkl', 'wb') as f:
        pickle.dump(gallery_avg, f)
        print("Average gallery (one embedding per person) saved to data/gallery_avg.pkl")

# W_norm = tf.nn.l2_normalize(loss_fn.W, axis=1)
# logits = tf.matmul(embeddings, W_norm, transpose_b=True)
# y_pred = tf.argmax(logits, axis=1).numpy()
