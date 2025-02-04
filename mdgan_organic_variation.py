from gan import Generator
from gan import Discriminator
from gan import GAN
from tensorflow.keras.models import load_model

def main():
  ''' This is a sample implementation of one generator training with three discriminators '''
  
  # Setting up the gan network system
  gan1 = GAN(generator=Generator(), discriminator=Discriminator(), nn=load_model("new_nn.h5"))
  gan2 = GAN(generator=gan1.G, discriminator=Discriminator(), nn=load_model("new_nn.h5"))
  gan3 = GAN(generator=gan1.G, discriminator=Discriminator(), nn=load_model("new_nn.h5"))
  
  # Set the number of training iterations for the network
  training_period = 5
  
  # Train the system
  for i in range(training_period):
    gan1.train(testid=i)
    gan2.train(testid=i)
    gan3.train(testid=i)
    
if __name__ == '__main__':
  main()
