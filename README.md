# OrGANic: Networking Generative Adversarial Networks

In this project, we network Generative Adversarial Networks(GANs) inspired by how trees in a forest
communicate with each other using mychorrhizal networks. The model is set up as a network of Generators and Discriminators
modelled as "trees"(each tree has its own generator and discriminator) which are "growing" in a "forest"(graph-like connection).
The "forest" model examines the evolutions of images, if any, as they pass from "tree" to "tree" with image quality, extent of 
mode collapse and morphing of image symbols being the primary aspects tested for. 

We developed a set of ten symbols for use as the training language(Refer to 'example pics' folder for samples). We used bezier curves(Refer to 'bezier.py' 
if interested) to achieve slight variations that resembled different handwritings. Once the network system was set up, we ran 
different variations of tests using our system. You can find the summarized results, as well as our conclusion, near the end of the readme.

### Prerequisites
1. You will need to have the modules pickle, tensorflow, numpy and keras installed. Refer to the source code to make sure
you do not get any module errors.

2. The forest's simulation time grows exponentially as the number of years increases due to introduction of new trees. If you are considering running the simulation
for a long period, ensure adequate computational power and memory or reduce the sprouting rate of trees(explained in the next section).

### Running a Forest Simulation
To run a forest simulation, you will need the following files:
  1. forest.py - main file
  2. gan.py
  3. tree.py
  4. graphing.py
  5. language_getter.py
  6. saveddekus/newdeku180.h5(or any other hdh5 file. This one, however, has the least occurrence of mode collapse among all our saved models)
  7. new_nn.h5 - neural network implementation

Before executing forest.py, you will need to create an object of the Forest class and call the grow function - an example is provided in the source code. The parameters
of the function are the rate - float - at which new trees will "sprout" and the years - integer - the forest should do its growing. The
rate is more of a probabilistic figure as opposed to a fixed number. For example, a rate of 1 means a high likelihood of 1 new tree every year. These parameters are arbitrary. For optimal performance and results, we recommend a rate of 1 and 10 years of growth.

### Images and Graphs Generated
During the forest simulation, images will be produced by individual trees in the forest. Additionally, after a year of training, a graph
is also produced to represent the current state of connections in the forest.

These images will be created in the current directory that hosts the source code files. If you wish to modify this, simply change the directory path 
in lines 70 in graphing.py and 205 in gan.py.

You can turn the image generation off by turning the plot attribute to False in forest.py - line 65.

### Results
1. Younger trees were worse teachers than older teachers. Images output from trees taught by 
younger trees were of significantly less quality than older ones. They also tended to display less mode collapse
given the inverse relationship between image quality and mode collapse.

2. We also found out that a variation of our networking model helped
to reduce mode collapse for GANs by networking multiple discriminators with a single generator. There does exist a variation of this, known as the mdgan. Our system, however, has its training done in a cyclical version from one discriminator to the next as opposed to
the mdgan's separate training for each discriminator. You can look at mdgan_organic_variation.py for a sample demonstration.

3. Morphing of image symbols,
which is the combination of two or more images into one, rarely occured (1 in 500 produced images). We could, however, not pin down
any conditions that facilitated the occurrence of morphing.

### Conclusion
We were able to conclude that GANs, in their current form, can not be used to mimic natural language evolution. Our project however managed to bring insight into how GANs can be networked on a massive scale.

### Acknowledgements
We would like to acknowledge Hamilton College Computer Science Department for giving us the chance to do this research and for all the 
resources that were availed to us. We would also like to acknowledge adaptation of code for our implementation of a Generative Adversarial 
Network from Datacamp By Stefan Hosein and code for our Neural Network Implementation Code from NextJournal by Gregor Koehler. We would
also like to acknowledge Github for giving us a platform to share our work and collaborate with others.

### Authors
The primary contributors to this research are Kenneth Talarico and Ian Nduhiu - Hamilton College Class of 2022, under the supervision
of Professor David Perkins, Computer Science Department. We welcome any contributions and suggestions to advance our project and the 
field of GANs.
