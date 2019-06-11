from PIL import Image
import numpy as np
import gan

def produce_language(generator, n=1):
    ''' Produces the language i.e. a 2d array of values ranging from 0 to 255
    '''
    all_generated_images = []
    for _ in range(n):
        noise = np.random.normal(0, 1, size=(100, 100))
        generated = generator.G.predict(noise)
        generated = generated.reshape(100, 28, 28)
        generated = (generated * 127.5) + 127.5
        generated = np.array(generated, dtype='int64')
        all_generated_images.append(generated)
    new_imgs = np.array(all_generated_images)
    new_imgs = new_imgs.reshape(new_imgs.shape[0] * new_imgs.shape[1],
                                new_imgs.shape[2], new_imgs.shape[3])
    return new_imgs

if __name__ == '__main__':
    gen1 = gan.Generator()
    my_gan = gan.GAN(generator=gen1)
    my_gan.train(epochs=10, plot=False)
    curr_xtrain = produce_language(gen1, 600)
    new_gan = gan.GAN(x_train=curr_xtrain)
    new_gan.train(epochs=5)
