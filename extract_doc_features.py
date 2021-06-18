def extract_doc_features(filepath: str) -> dict:
    """
    Функция, которая будет вызвана для получения признаков документа, для которого
    задан:
    :param filepath: абсолютный путь до тестового файла на локальном компьютере (строго
    pdf или png).

    :return: возвращаемый словарь, совпадающий по составу и написанию ключей условию
    задачи
    """
    # ваша реализация функции ниже

    import cv2
    import pytesseract
    import numpy as np
    import re
    from pytesseract import Output

    def get_main_text(img) -> str:
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['level'])

        text_main = ""
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            if w*h > 1000000:
                continue

            if d['level'][i] == 2 and text_main == "":
                if x > 10 and y > 10:
                    txt_main_img = img[-10 + y: y + h + 5, -10 + x: x + w + 5]
                else:
                    txt_main_img = img[y: y + h + 5, x: x + w]

                text_main = pytesseract.image_to_string(txt_main_img, lang = 'rus')

                if text_main.count('\n') > 3 or len(re.sub('[^А-Яа-я0-9 ]+', '', text_main)) < 4:
                    text_main = ""

        if text_main == "":
             text = pytesseract.image_to_string(img, lang = 'rus')
             text_main = text[ : text.find("\n") + 1]

        text_main = re.sub('[^А-Яа-я0-9 ]+', '', text_main[ : text_main.find("\n") + 1])

        return text_main

    def get_text_block(img) -> str:
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['level'])

        text_block = ""
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            if w*h < 100000:
                continue

            if d['level'][i] == 2 and text_block == "":
                if x > 10 and y > 10:
                    txt_block_img = img[-10+y: y + h + 40, -10+x: x + w + 40]
                else:
                    txt_block_img = img[y: y + h + 40, x: x + w + 40]

                text_block_unity = pytesseract.image_to_string(txt_block_img, lang = 'rus')
                text_block = text_block_unity.replace("\n", " ").split(' ')

                if len(text_block) < 20 or text_block_unity.count('\n') > len(text_block)/6:
                    text_block = ""

        text = ""

        for i, word in zip(range(len(text_block)), text_block):
            text += re.sub('[^А-Яа-я0-9 -]+', '', word)
            if i == 9:
                break
            else:
                text += ' '

        text_block = text

        return text_block

    def hide_table(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_lower = np.array((0, 0, 0), np.uint8)
        hsv_higher = np.array((255, 255, 180), np.uint8)
        blackText  = cv2.GaussianBlur(cv2.inRange(hsv, hsv_lower, hsv_higher), (5, 5), 0)

        imgBGR = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 1), np.uint8)
        imgBGR = cv2.dilate(imgBGR, kernel, iterations=1)
        imgBGR = cv2.erode(imgBGR, kernel, iterations=1)


        contours, hierarchy = cv2.findContours(blackText, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for currentContour, currentHierarchy in zip(contours, hierarchy[0]):
            approx = cv2.approxPolyDP(currentContour, 0.02*cv2.arcLength(currentContour, True), True)
            area = cv2.contourArea(currentContour)

            if area > 5500 and len(approx) == 4:
                x, y, w, h  = cv2.boundingRect(currentContour)
                cv2.rectangle(imgBGR, (x,y), (x+w,y+h), (255, 255, 255), -1)

        for (x, y, w, h) in position_of_color_objects(img, [0, 2, 100], [255, 255, 255]):
            cv2.rectangle(imgBGR, (x, y), (x+w,y+h), (0, 0, 0), -1)

        return imgBGR

    def position_of_color_objects(img, color_lower, color_higher) -> int:
        color_lower =  np.array([color_lower])
        color_higher = np.array([color_higher])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        color = cv2.dilate(cv2.inRange(hsv, color_lower, color_higher), np.ones((10, 10), "uint8"))
        selection = cv2.bitwise_and(img, img, mask = color)

        gray = cv2.cvtColor(selection, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        xywh = [(0, 0, 0, 0)]
        try:
            for currentContour, currentHierarchy in zip(contours, hierarchy[0]):
                area = cv2.contourArea(currentContour)
                x, y, w, h  = cv2.boundingRect(currentContour)

                if currentHierarchy[2] < 0 or area < 4000:
                    pass
                elif currentHierarchy[3] < 0:
                    xywh.append((x, y, w, h))
        except:
            pass

        return xywh

    def color_areas_count(img, color_lower, color_higher) -> int:
        color_lower =  np.array([color_lower])
        color_higher = np.array([color_higher])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        kernal = np.ones((10, 10), "uint8")
        color = cv2.dilate(cv2.inRange(hsv, color_lower, color_higher), kernal)
        selection = cv2.bitwise_and(img, img, mask = color)

        gray = cv2.cvtColor(selection, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        areas_count = 0

        try:
            for currentContour, currentHierarchy in zip(contours, hierarchy[0]):
                area = cv2.contourArea(currentContour)

                if currentHierarchy[2] < 0 or area < 4000:
                    pass
                elif currentHierarchy[3] < 0 and area > 4000:
                    areas_count += 1
        except:
            pass

        return areas_count

    def cells_detect_table(img):
        import random

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_min = np.array((0, 0, 0), np.uint8)
        hsv_max = np.array((255, 255, 180), np.uint8)
        thresh =  cv2.GaussianBlur(cv2.inRange(hsv, hsv_min, hsv_max), (3, 3), 0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cellIndex = 0

        for currentContour, currentHierarchy in zip(contours, hierarchy[0]):
            approx = cv2.approxPolyDP(currentContour, 0.01* cv2.arcLength(currentContour, True), True)
            area = cv2.contourArea(approx)

            if area > 1500 and (len(approx) <= 15) and currentHierarchy[3] > 0:
                cellIndex += 1


        return cellIndex

    img = cv2.imread(filepath)
    red_areas_count = color_areas_count(img, [136,86,0], [180,255,255])
    blue_areas_count = color_areas_count(img, [80, 30, 30], [140, 250, 250])
    text_main_title = get_main_text(hide_table(img))
    text_block = get_text_block(hide_table(img))
    table_cells_count = cells_detect_table(img)

    result = {
        'red_areas_count': red_areas_count, # количество красных участков (штампы, печати и т.д.) на скане
        'blue_areas_count': blue_areas_count, # количество синих областей (подписи, печати, штампы) на скане
        'text_main_title': text_main_title, # текст главного заголовка страницы или ""
        'text_block': text_block, # текстовый блок параграфа страницы, только первые 10 слов, или ""
        'table_cells_count': table_cells_count, # уникальное количество ячеек (сумма количеств ячеек одной или более таблиц)
    }

    return result
