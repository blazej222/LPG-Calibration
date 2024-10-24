import pytesseract
from datetime import timedelta,datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 0, 1, 1, 1): 9,
    # (0, 0, 0, 0, 0, 1, 1): '-'
}
H_W_Ratio = 1.9
THRESHOLD = 80
arc_tan_theta = 6.0  # 数码管倾斜角度

video_path = 'data/videos/2024-09-20 15-21-39.mkv'
output_file_path = 'data/output/ocr-output/2024-09-20 15-21-39-ocr-output.txt'

time_str = video_path.split('/')[-1].split('.')[0].split()[1]
time_format = "%H-%M-%S"
time_obj = datetime.strptime(time_str, time_format)

midnight = datetime.combine(time_obj.date(), datetime.min.time())  # Ustalamy północ jako punkt odniesienia
seconds_since_midnight = (time_obj - midnight)

# print(seconds_since_midnight)
# input()

detect_digit_primary_dimension = 4 # how many pixels must be in column to detect horizontal pixel
detect_digit_secondary_dimension = 4 # how many consecutive detected pixels must be next to another to start
# detecting digit TODO: Might not be necessary



def load_image(image, show=False):
    # todo: crop image and clear dc and ac signal
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_img.shape
    # crop_y0 = 0 if h <= crop_y0_init else crop_y0_init
    # crop_y1 = h if h <= crop_y1_init else crop_y1_init
    # crop_x0 = 0 if w <= crop_x0_init else crop_x0_init
    # crop_x1 = w if w <= crop_x1_init else crop_x1_init
    # gray_img = gray_img[crop_y0:crop_y1, crop_x0:crop_x1]
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    if show:
        cv2.imshow('gray_img', gray_img)
        cv2.imshow('blurred_img', blurred)
        cv2.waitKey(0)
    return blurred, gray_img


def preprocess(img, threshold, show=False, kernel_size=(5, 5)):
    # 直方图局部均衡化
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    img = clahe.apply(img)
    # 自适应阈值二值化
    dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, threshold)
    # 闭运算开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)

    if show:
        cv2.imshow('equlizeHist', img)
        cv2.imshow('threshold', dst)
        cv2.waitKey(0)
    return dst


def helper_extract(one_d_array, threshold=20):
    res = []
    flag = 0
    temp = 0
    # Let's assume we're checking horizontally (summing by column)
    for i in range(len(one_d_array)):
        if one_d_array[i] < detect_digit_primary_dimension * 255: # if there's not enough pixels in this column
            if flag > threshold: # check if there have been enough pixels in a row
                start = i - flag
                end = i
                temp = end
                if end - start > detect_digit_secondary_dimension: #TODO: Is this even necessary?
                    res.append((start, end))
            flag = 0
        else:
            flag += 1

    else:
        if flag > threshold:
            start = temp
            end = len(one_d_array)
            if end - start > 50:
                res.append((start, end))
    return res


def find_digits_positions(img, reserved_threshold=20):
    # cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # digits_positions = []
    # for c in cnts[1]:
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 0), 2)
    #     cv2.imshow('test', img)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow('test')
    #     if w >= reserved_threshold and h >= reserved_threshold:
    #         digit_cnts.append(c)
    # if digit_cnts:
    #     digit_cnts = contours.sort_contours(digit_cnts)[0]

    digits_positions = []
    img_array = np.sum(img, axis=0)
    horizon_position = helper_extract(img_array, threshold=4)
    img_array = np.sum(img, axis=1)
    vertical_position = helper_extract(img_array, threshold=4)
    # make vertical_position has only one element
    if len(vertical_position) > 1:
        vertical_position = [(vertical_position[0][0], vertical_position[len(vertical_position) - 1][1])]
    for h in horizon_position:
        for v in vertical_position:
            digits_positions.append(list(zip(h, v)))
    if len(digits_positions) > 0:
        return digits_positions
    else:
        return None


def recognize_digits_line_method(digits_positions, output_img, input_img):
    digits = []
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = input_img[y0:y1, x0:x1]
        h, w = roi.shape
        suppose_W = max(1, int(h / H_W_Ratio))

        # Eliminate irrelevant symbol interference
        # if x1 - x0 < 25 and cv2.countNonZero(roi) / ((y1 - y0) * (x1 - x0)) < 0.2:
        #     continue

        # Identify case 1 separately
        if w < suppose_W / 2:
            x0 = max(x0 + w - suppose_W, 0)
            roi = input_img[y0:y1, x0:x1]
            w = roi.shape[1]

        center_y = h // 2
        quater_y_1 = h // 4
        quater_y_3 = quater_y_1 * 3
        center_x = w // 2
        line_width = 5  # line's width
        width = (max(int(w * 0.15), 1) + max(int(h * 0.15), 1)) // 2
        small_delta = int(h / arc_tan_theta) // 4
        segments = [
            ((w - 2 * width, quater_y_1 - line_width), (w, quater_y_1 + line_width)),
            ((w - 2 * width, quater_y_3 - line_width), (w, quater_y_3 + line_width)),
            ((center_x - line_width - small_delta, h - 2 * width), (center_x - small_delta + line_width, h)),
            ((0, quater_y_3 - line_width), (2 * width, quater_y_3 + line_width)),
            ((0, quater_y_1 - line_width), (2 * width, quater_y_1 + line_width)),
            ((center_x - line_width, 0), (center_x + line_width, 2 * width)),
            ((center_x - line_width, center_y - line_width), (center_x + line_width, center_y + line_width)),
        ]
        on = [0] * len(segments)

        for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
            seg_roi = roi[ya:yb, xa:xb]
            # plt.imshow(seg_roi, 'gray')
            # plt.show()
            total = cv2.countNonZero(seg_roi)
            area = (xb - xa) * (yb - ya) * 0.9
            #print('prob: ', total / float(area))
            if total / float(area) > 0.35:
                on[i] = 1
        # print('encode: ', on)
        if tuple(on) in DIGITS_LOOKUP.keys():
            digit = DIGITS_LOOKUP[tuple(on)]
            digits.append(digit)
        else:
            digit = '*'

        #digits.append(digit)

        # Decimal point recognition
        #print('dot signal: ',cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9 / 16 * width * width))
        if cv2.countNonZero(roi[h - int(3 * width / 4):h, w - int(3 * width / 4):w]) / (9. / 16 * width * width) > 0.5 and digit == '*':
            digits.append(',')
            cv2.rectangle(output_img,
                          (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4)),
                          (x1, y1), (0, 128, 0), 2)
            cv2.putText(output_img, 'dot',
                        (x0 + w - int(3 * width / 4), y0 + h - int(3 * width / 4) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)

        cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
        cv2.putText(output_img, str(digit), (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)
    return digits


# Funkcja do ekstrakcji tekstu z określonego regionu
def extract_text_from_frame(frame, roi):
    x, y, w, h = roi
    roi_frame = frame[y:y + h, x:x + w]

    blurred, gray_img = load_image(roi_frame, show=False)
    output = blurred
    dst = preprocess(blurred, THRESHOLD, show=False)

    #cv2.imwrite("test.png", dst)

    digits_positions = find_digits_positions(dst)

    #print(digits_positions)

    if digits_positions is None: return
    digits = recognize_digits_line_method(digits_positions, output, dst)

    #print(digits)

    # cv2.imshow('output', output)
    # cv2.waitKey()
    #cv2.destroyAllWindows()

    return digits

# Wczytaj wideo
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 15 # co jedną sekundę
frame_count = 0

# Definicje obszarów do ekstrakcji tekstu (x, y, szerokość, wysokość)
roi_pole1 = (55, 1520, 341, 120)
roi_pole2 = (415, 1520, 341, 120)
roi_pole3 = (775, 1520, 341, 120)
roi_pole4 = (1160, 1520, 341, 120)
roi_pole5 = (1555, 1520, 341, 120)

last_timestamp_min = 0

# Otwórz plik do zapisu
with open(output_file_path, 'w') as file:
    #file.write('Czas RPM     Benzyna    Gaz     Podcisnienie    Cisnienie\n')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0 and frame_count >= 120:
            time_stamp = timedelta(seconds=int(frame_count // fps),milliseconds=int((frame_count % fps)* (1000 / fps)))

            # cv2.imshow("Frame",frame)

            # Ekstrakcja tekstu z poszczególnych pól
            pole1 = extract_text_from_frame(frame, roi_pole1)
            pole2 = extract_text_from_frame(frame, roi_pole2)
            pole3 = extract_text_from_frame(frame, roi_pole3)
            pole4 = extract_text_from_frame(frame, roi_pole4)
            pole5 = extract_text_from_frame(frame, roi_pole5)

            shouldSave = True

            if pole1 is not None:
                pole1 = ''.join(str(e) for e in pole1)
            else:
                pole1 = 'x'
                shouldSave = False

            if pole2 is not None:
                pole2 = ''.join(str(e) for e in pole2)
            else:
                pole2 = 'x'
                shouldSave = False

            if pole3 is not None:
                pole3 = ''.join([str(e) for e in pole3])
            else:
                pole3 = 'x'
                shouldSave = False

            if pole4 is not None:
                pole4 = ''.join([str(e) for e in pole4])
            else:
                pole4 = 'x'
                shouldSave = False

            if pole5 is not None:
                pole5 = ''.join([str(e) for e in pole5])
            else:
                pole5 = 'x'
                shouldSave = False

            # Zapis wyników do pliku

            time_write = (time_stamp+seconds_since_midnight).total_seconds()

            if shouldSave: file.write(f"{time_write};{pole1};{pole2};{pole3};{pole4};{pole5}\n")
            print(time_write)
            # if time_stamp.seconds != last_timestamp_min and time_stamp.seconds % 60:
            #     print(time_stamp.min)
            # last_timestamp_min = time_stamp.min

        frame_count += 1

# Zwolnij zasoby
file.close()
cap.release()
cv2.destroyAllWindows()
