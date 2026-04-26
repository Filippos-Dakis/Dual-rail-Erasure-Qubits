import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# colors
blue   = np.array([0.1215686, 0.4666667, 0.7058824])  # custom blue
green =  np.array([0.1801333333, 0.7176571429, 0.6424333333])
orange = np.array([1.0, 0.49803921568, 0.05490196078])   # target orange

# number of interpolation steps per section
n_blue_green   = 60
n_green_orange = 60

# interpolate blue -> green
blue_to_green = np.linspace(blue, green, n_blue_green, endpoint=False)

# interpolate green -> orange
green_to_orange = np.linspace(green, orange, n_green_orange)

# combine
custom_map_1 = np.vstack((blue_to_green, green_to_orange))

# make matplotlib colormap
cmap_1 = LinearSegmentedColormap.from_list("blue_green_orange", custom_map_1)

# -----------------------------------------------------------------------------

# colors
black_vt = np.array([0., 0., 0.])  
green_vt = np.array([0.112, 150/250, 114/250])
pink_vt  = np.array([0.949, 0.517, 0.623])   

# number of interpolation steps per section
n_black_green = 60
n_green_pink  = 60

# interpolate blue -> green
black_to_green = np.linspace(black_vt, green_vt, n_black_green, endpoint=False)

# interpolate green -> orange
green_to_pink = np.linspace(green_vt, pink_vt, n_green_pink)

# combine
custom_map_2 = np.vstack((black_to_green, green_to_pink))

# make matplotlib colormap
cmap_2 = LinearSegmentedColormap.from_list("black_green_pink", custom_map_2)

# ------------------------------------------------------------------------------


# test plot
#gradient = np.linspace(0, 1, 256).reshape(1, -1)
#plt.figure(figsize=(6, 2))
#plt.imshow(gradient, aspect='auto', cmap=cmap_3)
#plt.axis("off")
#plt.show()