import numpy as np
import nibabel as nib
import os
import random

# Define function to load data from file
def load_data(path):
    # Load the MRI images
    t1_img = nib.load(os.path.join(path, 't1.nii.gz')).get_fdata()
    t1ce_img = nib.load(os.path.join(path, 't1ce.nii.gz')).get_fdata()
    t2_img = nib.load(os.path.join(path, 't2.nii.gz')).get_fdata()
    flair_img = nib.load(os.path.join(path, 'flair.nii.gz')).get_fdata()

    # Load the tumor segmentations
    seg_img = nib.load(os.path.join(path, 'seg.nii.gz')).get_fdata()

    # Convert the tumor segmentations to binary masks
    seg_mask = np.zeros_like(seg_img)
    seg_mask[seg_img == 1] = 1  # Necrotic and non-enhancing tumor
    seg_mask[seg_img == 2] = 1  # Edema
    seg_mask[seg_img == 4] = 1  # Enhancing tumor

    # Normalize the pixel values to a range between 0 and 1
    t1_norm = (t1_img - np.min(t1_img)) / (np.max(t1_img) - np.min(t1_img))
    t1ce_norm = (t1ce_img - np.min(t1ce_img)) / (np.max(t1ce_img) - np.min(t1ce_img))
    t2_norm = (t2_img - np.min(t2_img)) / (np.max(t2_img) - np.min(t2_img))
    flair_norm = (flair_img - np.min(flair_img)) / (np.max(flair_img) - np.min(flair_img))

    # Combine the images into a single array
    image = np.stack((t1_norm, t1ce_norm, t2_norm, flair_norm), axis=-1)

    return image, seg_mask

# Define function to load data from multiple files
def load_data_from_files(path_list):
    images = []
    masks = []
    for path in path_list:
        image, mask = load_data(path)
        images.append(image)
        masks.append(mask)
    return np.array(images), np.array(masks)

# Load the data from the BraTS dataset
train_path = 'path/to/train/folder'
val_path = 'path/to/val/folder'
test_path = 'path/to/test/folder'

train_images, train_masks = load_data_from_files(os.listdir(train_path))
val_images, val_masks = load_data_from_files(os.listdir(val_path))
test_images, test_masks = load_data_from_files(os.listdir(test_path))

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# Define function to perform data augmentation
def augment_data(image, mask):
    # Randomly flip the image and mask horizontally
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    # Randomly flip the image and mask vertically
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)

    # Randomly rotate the image and mask
    k = random.randint(0, 3)
        mask = np.rot90(mask, k=k)
    image = np.rot90(image, k=k)

    # Apply elastic deformation to the image and mask
    alpha = random.uniform(100, 300)
    sigma = random.uniform(8, 12)
    random_state = np.random.RandomState(None)
    shape = image.shape[:-1]
    dx = np.random.normal(0, sigma, shape)
    dy = np.random.normal(0, sigma, shape)
    dz = np.random.normal(0, sigma, shape)
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z+dz, (-1, 1))
    image = ndimage.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape+(4,))
    mask = ndimage.map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape+(1,))

    return image, mask

# Define the U-Net model
def unet_model(input_size=(240, 240, 4)):
    inputs = layers.Input(input_size)

    # Contracting path
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottom
    conv5 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    # Expansive path
    up6 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2, 2))(drop5))

    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = layers.Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    return model

# Define the Dice coefficient loss function
def dice_coef_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Define the Dice coefficient metric
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Define the data generator for training and validation
def data_generator(images, masks, batch_size=32, shuffle=True, augment=True):
    while True:
        if shuffle:
            idx = np.random.permutation(images.shape[0])
            images = images[idx]
            masks = masks[idx]
        
            batch_masks = masks[i:i+batch_size]

            # Augment the data
            if augment:
                batch_images, batch_masks = zip(*[augment_data(image, mask) for image, mask in zip(batch_images, batch_masks)])

            # Convert the data to arrays
            batch_images = np.array(batch_images)
            batch_masks = np.array(batch_masks)

            yield batch_images, batch_masks

# Train the model
model = unet_model()
model.fit(data_generator(train_images, train_masks), steps_per_epoch=len(train_images)//32, epochs=50, validation_data=data_generator(val_images, val_masks, augment=False), validation_steps=len(val_images)//32)

# Evaluate the model
eval_metrics = model.evaluate(test_images, test_masks)
print("Evaluation metrics: ", eval_metrics)
