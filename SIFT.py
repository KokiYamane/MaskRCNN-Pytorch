import cv2


def main(args):
    img = cv2.imread(args.image)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_sift = cv2.drawKeypoints(img, keypoints, None, flags=4)
    cv2.imwrite(args.output, img_sift)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--output', type=str, default='./results/sift_img.jpg')
    args = parser.parse_args()
    main(args)
