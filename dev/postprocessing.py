def crop_image(image, boxes):
    # calculate the crop size based on the bounding box
    x_min, y_min = boxes.numpy()[:, 0:2].min(axis=0)
    x_max, y_max = boxes.numpy()[:, 2:4].max(axis=0)
    image = image.crop((x_min, y_min, x_max, y_max))
    return image