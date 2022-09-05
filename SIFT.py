import cv2


def main(args):
    # img = cv2.imread(args.image)
    # sift = cv2.xfeatures2d.SIFT_create()
    # keypoints, descriptors = sift.detectAndCompute(img, None)
    # img_sift = cv2.drawKeypoints(img, keypoints, None, flags=4)
    # cv2.imwrite(args.output, img_sift)

    image_base = cv2.imread(args.image_base)
    image = cv2.imread(args.image)
    # orb = cv2.ORB_create()

    # kp1, des1 = orb.detectAndCompute(image_base, None)
    # kp2, des2 = orb.detectAndCompute(image, None)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # matches = bf.match(des1, des2)

    # matches = sorted(matches, key=lambda x: x.distance)

    # matches = cv2.drawMatches(
    #     image_base, kp1, image, kp2, matches[:25], None, flags=2)
    # cv2.imwrite(args.output, matches)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image_base, None)
    kp2, des2 = sift.detectAndCompute(image, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for match1, match2 in matches:
        if match1.distance < 0.75 * match2.distance:
            good.append([match1])

    sift_matches = cv2.drawMatchesKnn(
        image_base, kp1, image, kp2, good, None, flags=2)

    cv2.imwrite(args.output, sift_matches)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--image_base', type=str)
    parser.add_argument('--output', type=str, default='./results/sift_img.jpg')
    args = parser.parse_args()
    main(args)
