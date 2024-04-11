import numpy as np
import cv2
from sklearn.cluster import KMeans


def get_dominant_colors(img_path,num_colors=1):
    img = cv2.imread(img_path)

    pixels = img.reshape(-1,3)

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

def get_all_colors(img_path,num_color):
    img = cv2.imread(img_path)

    pixels = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_color)
    kmeans.fit(pixels)

    cluster_centers = kmeans.cluster_centers_.astype(int)
    return cluster_centers

def display_colors(colors):
    height =200
    width = len(colors)*300
    color_display = np.zeros((height,width, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        color_display[:, i * 200:(i + 1) * 300] = color


    cv2.imshow('Dominant Colors', color_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_all_colors(colors):
    height = 200
    width = len(colors)*100
    display = np.zeros((height,width, 3), dtype=np.uint8)

    for i, color in enumerate(colors):
        display[:, i * 100:(i + 1) * 100] = color



    cv2.imshow('all Colors',display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    img_path = "sample.jpeg"

    num_colors = 1
    num_color=9

    dominant_colors = get_dominant_colors(img_path, num_colors)
    display_colors(dominant_colors)

    all_colors = get_all_colors(img_path,num_color)
    display_all_colors(all_colors)

    print("Dominant color:",dominant_colors)
    print("All colors RGB format:", all_colors)
