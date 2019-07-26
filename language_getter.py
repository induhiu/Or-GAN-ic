''' Implementation by Ian Nduhiu. Gets a generator to produce images in bulk
    for reintroduction into a forest(system of networked GANs) or prediction
    using a neural network. Default images produced are in the shape of
    (60000, 28, 28) '''
import numpy as np

def produce_language(gen, n=600):
    ''' Produces the language i.e. a 2d array of values ranging from 0 to 255.
    Takes a generator(the seq model, not the object) and optional n as
    parameters and returns a numpy array of n * 100 images  '''
    all_generated_images = []
    for i in range(n):
        noise = np.random.normal(0, 1, size=(100, 100)) # noise for generator
        generated = gen.predict(noise)  # produces 100 images
        generated = generated.reshape(100, 28, 28)
        generated = (generated * 127.5) + 127.5  # brings values to range 0-255
        generated = np.array(generated, dtype='int64')
        all_generated_images.append(generated)

    new_imgs = np.array(all_generated_images)
    # Reshape to take form of (n * 100, 28, 28)
    new_imgs = new_imgs.reshape(new_imgs.shape[0] * new_imgs.shape[1],
                                new_imgs.shape[2], new_imgs.shape[3])
    return new_imgs
