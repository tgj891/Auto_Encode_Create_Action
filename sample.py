import cv2

if __name__ == '__main__':
    # cap = cv2.VideoCapture(r"C:\Users\liev\Desktop\myproject\move.Mp4")
    cap = cv2.VideoCapture(r"C:\Users\liev\Desktop\data\move\shot.Mp4")
    i = 0
    count = 1
    while cap.isOpened():
        ret, fram = cap.read()
        i += 1
        if i < 2:
            continue
        img = fram[0:480, 186:666, :]
        cv2.imwrite(r"C:\Users\liev\Desktop\data\move\old\%d.jpg"%count, img)
        count += 1
        cv2.imshow("img", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
