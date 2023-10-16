from dev.utils import cal_max_bb
def crop_image(image, boxes):
    # calculate the crop size based on the bounding box
    x_min, y_min, x_max, y_max = cal_max_bb(boxes)
    image = image.crop((x_min, y_min, x_max, y_max))
    return image