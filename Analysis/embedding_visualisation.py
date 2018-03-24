import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from skimage.transform import resize

from tensorflow.contrib.tensorboard.plugins import projector

import FileManager


def create_sprite_image(images, masks):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)

    # combine with mask
    images = np.array([resize(np.squeeze(im), output_shape=(28, 28)) for im in images])
    masks  = np.array([resize(np.squeeze(ms), output_shape=(28, 28)) for ms in masks])
    images = images*(0.2+0.8*masks)

    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def create_metadata(labels, path):
    with open(path, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("%d\t%d\n" % (index, label))


def create_embedding(embed, name='Embedding', path = 'E:/logs/', label = ''):
    embedding_var = tf.Variable(embed, name=name)
    summary_writer = tf.summary.FileWriter(path)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the metadata
    embedding.metadata_path = path + 'metadata{}.tsv'.format('_' + label)
    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = path + 'sprite{}.png'.format('_'+label)
    embedding.sprite.single_image_dim.extend([28, 28])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join('E:/logs/', 'model{}.ckpt'.format('_'+label)), 1)



if __name__ == "__main__":

    # dir103, siam100
    run = '100'
    net_type = 'siam'

    #load data
    #Embed = FileManager.Embed('siamQ')
    #images_v, embed_v, meta_v, labels_v, masks_v = Embed.load('078X', 24, 'Valid')
    #images_t, embed_t, meta_t, labels_t, masks_t = Embed.load('078X', 24, 'Train')

    Embed = FileManager.Embed(net_type)
    #images_v, embed_v, meta_v, labels_v, masks_v = Embed.load(name, 16, 'Valid')
    #images_t, embed_t, meta_t, labels_t, masks_t = Embed.load(name, 16, 'Train')
    images_t, embed_t, meta_t, labels_t, masks_t = Embed.load(run, 40, 'Test')

    #images = np.concatenate([images_v, images_t])
    #embed  = np.concatenate([embed_v, embed_t])
    #labels = np.concatenate([labels_v, 2+labels_t])
    #masks  = np.concatenate([masks_v, masks_t])
    #meta = meta_v + meta_t
    images = images_t
    embed = embed_t
    labels = labels_t
    masks = masks_t
    meta = meta_t

    print("Loaded Embedding Data")

    # Create Embedding
    create_embedding(embed, net_type+run)
    print("Created Embeddings..")

    # Create Sprite
    sprite_image = create_sprite_image(images, masks)
    plt.imsave('E:/logs/sprite_{}{}.png'.format(net_type, run), sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')
    print("Created Sprite..")

    #create meta
    create_metadata(labels, 'E:/logs/metadata_{}{}.tsv'.format(net_type, run))
    print("Created Metadata..")

    print("Done!")