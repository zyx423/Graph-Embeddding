# GAE-and-VGAE
This is a code file containing the graph autoencoders (GAE) and variational graph autoencoder （VAE).
I must emphasize that the latent representations learned by the VGAE can't directly be used for tasks,  e.g, clustering, and classification.

Even though many bloggers think that VGAE can be used to learn the latent representations that retain the samples information in low-dimension space, but I think the purpose of this is to generate new samples, not to reduce data dimension.
