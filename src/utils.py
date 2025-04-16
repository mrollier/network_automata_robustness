from matplotlib.colors import ListedColormap

def get_ugent_colors_dict():
    ugent_colors = {
    'UGent Blue': '#1E64C8',
    'UGent Yellow': '#FFD200',
    'White': '#FFFFFF',
    'Black': '#000000',
    'LW': '#F1A42B',  # Faculty of Arts and Philosophy
    'RE': '#DC4E28',  # Faculty of Law and Criminology
    'WE': '#2D8CA8',  # Faculty of Sciences
    'GE': '#E85E71',  # Faculty of Medicine and Health Sciences
    'EA': '#8BBEE8',  # Faculty of Engineering and Architecture
    'EB': '#AEB050',  # Faculty of Economics and Business Administration
    'DI': '#825491',  # Faculty of Veterinary Medicine
    'PP': '#FB7E3A',  # Faculty of Psychology and Educational Sciences
    'BW': '#27ABAD',  # Faculty of Bioscience Engineering
    'FW': '#BE5190',  # Faculty of Pharmaceutical Sciences
    'PS': '#71A860'   # Faculty of Political and Social Sciences
    }
    return ugent_colors

def get_ugent_cmap():
    ugent_colors_dict = get_ugent_colors_dict()
    ugent_white = ugent_colors_dict['White']
    ugent_blue = ugent_colors_dict['UGent Blue']
    cmap = ListedColormap([ugent_white, ugent_blue])  # 0: white, 1: blue
    return cmap

def get_heart_shaped_grid():
    """
    Generates a heart-shaped grid of points.
    Returns a list of tuples representing the coordinates of the points in the grid.
    """
    heart_shape = []
    for x in range(-10, 11):
        for y in range(-10, 11):
            if (x**2 + (y - (x**2)**0.5)**2 <= 1):
                heart_shape.append((x, y))
    return heart_shape