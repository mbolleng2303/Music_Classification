


import numpy as np



def generate_random_data():
    N_images = 100
    size = [128, 128]
    features = 3
    N_classes = 3

    img = np.random.rand(N_images, features, size[0], size[1])
    label = np.random.randint(0, N_classes, N_images)

    img.tofile('img.csv', sep=',')
    label.tofile('label.csv', sep=',')



generate_random_data()

