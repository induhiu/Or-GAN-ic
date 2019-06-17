''' Gets language from a gan to pass to another gan. Implementation and
documentation by Ian Nduhiu '''

# from PIL import Image
import numpy as np
import gan
import sys

def produce_language(gen, n=600):
    ''' Produces the language i.e. a 2d array of values ranging from 0 to 255.
    Takes a generator(the seq model, not the object) and optional n as
    parameters and returns a numpy array of n * 100 images  '''
    # noise = np.random.normal(0, 1, size=(100, 100))
    all_generated_images = []
    for i in range(n):
        noise = np.random.normal(0, 1, size=(100, 100))
        # # for debugging if you want to
        # if i % 100 == 0:
        #     generated = gan.plot_generated_images(id=i, generator=gen.G)
        # else:
        #     generated = gen.G.predict(noise)
        # # Comment out the line below if you intend to use the debugging code
        # # above
        generated = gen.predict(noise)
        generated = generated.reshape(100, 28, 28)
        generated = (generated * 127.5) + 127.5
        generated = np.array(generated, dtype='int64')
        all_generated_images.append(generated)
        # for img in generated:
        #     all_generated_images.append(img)
    new_imgs = np.array(all_generated_images)
    new_imgs = new_imgs.reshape(new_imgs.shape[0] * new_imgs.shape[1],
                                new_imgs.shape[2], new_imgs.shape[3])
    return new_imgs

# if __name__ == '__main__':
#     xtrain, xtest = slicedata(np.load('imgarys.npz'), 60000)
#     gen1 = gan.Generator()
#     my_gan = gan.GAN(generator=gen1, x_train=xtrain, x_test=xtest)
#     my_gan.train(epochs=15, plot=False)
#     curr_xtrain = produce_language(gen1, 600)
#
#     # # If you want to see one image
#     # img = Image.fromarray(curr_xtrain[0].astype('uint8'))
#     # img.save('test.png')
#
#     gen2 = gan.Generator()
#     disc = gan.Discriminator()
#     new_gan = gan.GAN(generator= gen2, discriminator=disc, x_train=curr_xtrain)
#     new_gan.train(epochs=8)
