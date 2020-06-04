class Args:
    num_workers = 1
    image_size = 28                   # Spatial size of training images
    num_c = 1                         # Number of channels in the training images. For color images this is 3
    dim_z = 100                       # Size of z latent vector (i.e. size of generator input)
    dim_gen_feature = 28              # Size of feature maps in generator
    dim_dis_feature = 28              # Size of feature maps in discriminator
    num_epochs = 10                   # Number of training epochs
    lr_G = 0.0002                     # Learning rate for optimizers G
    lr_D = 0.0001                     # Learning rate for optimizers D
    beta1 = 0.5                       # Beta1 hyperparam for Adam optimizers
    num_gpu = 1                       # Number of GPUs available. Use 0 for CPU mode.
