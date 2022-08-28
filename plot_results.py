import numpy as np
import cv2


def CIERGB2RGB(xyz: np.ndarray):
    A = np.array([
        [3.134187, -1.617209, -0.490694],
        [-0.978749, 1.916130, 0.033433],
        [0.071964, -0.228994, 1.405754],
    ])
    rgb = np.dot(A, xyz)
    rgb = np.clip(rgb, 0, 1)
    return rgb


def CIELAB2CIERGB(Lab: np.ndarray):
    L, a, b = Lab
    f_y = (L + 16) / 116
    f_x = a / 500 + f_y
    f_z = f_y - b / 200
    Y = f_y ** 3 if f_y ** 3 > 0.008856 else (f_y - 16 / 116) / 7.787
    X = f_x ** 3 if f_x ** 3 > 0.008856 else (f_x - 16 / 116) / 7.787
    Z = f_z ** 3 if f_z ** 3 > 0.008856 else (f_z - 16 / 116) / 7.787
    return X, Y, Z


def CIELAB2RGB(Lab: np.ndarray):
    X, Y, Z = CIELAB2CIERGB(Lab)
    rgb = CIERGB2RGB([X, Y, Z])
    return rgb


def gradient_color(
    h1: float,
    h2: float,
    num: int,
    C: float = 40,
    L: float = 85,
):
    theta1 = np.pi / 180 * h1
    theta2 = np.pi / 180 * h2
    ab = [(np.cos(theta), np.sin(theta))
          for theta in np.linspace(theta1, theta2, num)]
    ab = C * np.array(ab)
    return [255 * CIELAB2RGB(np.array([L, a, b])) for a, b in ab]


def plot_segmentation_masks(fig, images, outputs, num_class=5, epoch=0):
    fig.clf()
    # row, col = 5, 10
    # row, col = 1, 5
    # col = 5
    # row = math.floor(len(images) / 5)
    # for i, (image, output) in enumerate(zip(images, outputs)):
    col = np.floor(np.sqrt(len(images))).astype(np.int)
    row = col

    # make color map
    key = 180
    width = 180
    color_map = gradient_color(key - width, key + width, num_class)
    # print(color_map)

    for i in range(col * row):
        ax = fig.add_subplot(row, col, i + 1)

        image = images[i]
        image = image.transpose(1, 2, 0)
        image = (255 * image).astype(np.uint8)
        image = cv2.UMat(image)

        output = outputs[i]
        masks = output['masks']
        scores = output['scores']
        label = output['labels']
        # print(masks)
        # print(np.sum(masks))
        # print(output['labels'])
        for mask, score, label in zip(masks, scores, label):
            if score < 0.75:
                continue

            mask = mask.transpose(1, 2, 0)
            mask = (255 * mask).astype(np.uint8)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            ret, mask = cv2.threshold(
                mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            # color_map = [
            #     (0, 0, 0),
            #     (255, 0, 0),
            #     (0, 255, 0),
            #     (0, 0, 255),
            #     (0, 255, 255),
            # ]

            contour = max(contours, key=lambda x: cv2.contourArea(x))
            cv2.drawContours(
                image,
                [contour],
                -1,
                # color=(0, 255, 0),
                # thickness=10,
                # color=(0, int(255 * score), 0),
                color=color_map[label],
                thickness=int(10 * score),
            )
        image = image.get()
        ax.imshow(image)
        ax.axis('off')

    fig.suptitle('{} epoch'.format(epoch))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
