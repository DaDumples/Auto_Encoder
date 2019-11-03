# Auto_Encoder

To run:

python my_testing.py

This was me messing around with autoencoding neural networks. An autoencoding neural network takes an input (eg,
an image from the MNIST fashion dataset) and outputs that same image. At some point in the network you bottleneck it to a very low
dimensional space, forcing the network to learn how to "encode" your data. This low dimensional space is known as the "latent space".

If you then estimate the covariance between each dimension of the latent space, you can find a best fit for N numbers of linearly
independant axes within this space. This is known as PCA and us reached by taking the eigenvalues of the covariance matrix.

my_testing.py maps 20 sliders to 20 linearly independant axes within the trained latent space. This user created vector is then fed
into the decoder and an image is generated. This implementation is very coarse, but if you mess with the top two sliders you should see
shoes, shirts, and pants generally materialize before you. I trained it for about 30 seconds before getting bored and only gave the network
about three layers, so I would say there is some serious room for improvement should I ever decide to get out of bed. Given that, I am
super satisfied with my results.

my_encoder.py trains the autoencoding neural network. my_testing.py runs the interactive GUI.
