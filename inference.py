import cv2
import numpy as np
import os
from face_aligment import extract_face
from model import *
import argparse
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys
import matplotlib.pyplot as plt

from PIL import Image

labels = {np.str_('14'): 0, np.str_('17'): 1, np.str_('25'): 2, np.str_('26'): 3,
         np.str_('34'): 4, np.str_('44'): 5, np.str_('45'): 6, np.str_('47'): 7,
         np.str_('49'): 8, np.str_('5'): 9, np.str_('63'): 10, np.str_('65'): 11,
         np.str_('67'): 12, np.str_('77'): 13,
         np.str_('80'): 14, np.str_('95'): 15, np.str_('99'): 16}

names = {v: k for k, v in labels.items()}



def load_image(path, return_face_only=True):
    image = cv2.imread(path)
    face, _, __ = extract_face(image)
    face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(face_gray, (128, 128))

    image_input = resized.reshape((128, 128, 1))
    image_input = np.concatenate([image_input] * 3, axis=-1)

    if return_face_only:
        return image_input[np.newaxis, ...]
    else:
        return image_input[np.newaxis, ...], face


def find_best_match(new_embedding, gallery_dict):
    # new_embedding = np.expand_dims(new_embedding, axis=0)
    best_match = None
    best_score = -1
    for label, emb in gallery_dict.items():
        score = cosine_similarity(new_embedding, np.expand_dims(emb, axis=0))[0][0]
        if score > best_score:
            best_score = score
            best_match = label
    return best_match, best_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()

    # Load and show image using PIL
    if os.path.exists(args.input):
        image = Image.open(args.input).convert('RGB')
        image = image.resize((512, 512))  # Optional: resize for better viewing
        image.show(title="Input Image")   # This will open in default image viewer
    else:
        print(f"Image not found: {args.input}")

    with open('data/gallery_avg.pkl', 'rb') as f:
        gallery_avg = pickle.load(f)

    n_classes = 17
    model = load_model(n_classes)
    model.load_weights("checkpoints/hamface_model.h5")
    print("Model loaded successfully.")

    loss_fn = HAMFaceLoss(num_classes=n_classes)
    class_weights = np.load("checkpoints/hamface_class_weights.npy")
    loss_fn.W.assign(class_weights)

    data, original_face = load_image(args.input, return_face_only=False)
    embeddings = model.predict([data, data])
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)

    match, score = find_best_match(embeddings, gallery_avg)
    print(f"\n\n *** Identity is ID: {names[match]}, Score: {score:.2f}")


    image_rgb = cv2.cvtColor(original_face, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(f"ID: {names[match]}, Score: {score:.2f}")
    plt.axis('off')
    plt.show()
