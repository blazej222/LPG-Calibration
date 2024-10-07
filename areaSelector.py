import cv2

# Globalne zmienne do przechowywania współrzędnych prostokąta
drawing = False  # True, jeśli przycisk myszy jest wciśnięty
ix, iy = -1, -1  # Początkowe współrzędne prostokąta
rectangles = []  # Lista do przechowywania wszystkich zaznaczonych obszarów
scale_factor_x = 1
scale_factor_y = 1

# Funkcja obsługi myszki
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, frame

    # Skaluje współrzędne myszy do rozdzielczości oryginalnej
    real_x = int(x / scale_factor_x)
    real_y = int(y / scale_factor_y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = real_x, real_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Tworzenie tymczasowego prostokąta na bieżącej klatce
            frame_copy = frame.copy()
            # Skaluje współrzędne z powrotem do rozdzielczości ekranu
            cv2.rectangle(frame_copy, (int(ix * scale_factor_x), int(iy * scale_factor_y)),
                          (x, y), (0, 255, 0), 2)
            cv2.imshow('Video', frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Rysowanie ostatecznego prostokąta na klatce
        cv2.rectangle(frame, (int(ix * scale_factor_x), int(iy * scale_factor_y)),
                      (x, y), (0, 255, 0), 2)
        rectangles.append((ix, iy, real_x - ix, real_y - iy))  # Przechowujemy (x, y, szerokość, wysokość)
        cv2.imshow('Video', frame)
        print(f"Zaznaczony obszar: {(ix, iy, real_x - ix, real_y - iy)}")

# Wczytaj wideo
video_path = '2024-09-20 15-21-39.mkv'
cap = cv2.VideoCapture(video_path)

# Oryginalne wymiary wideo
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Rozdzielczość ekranu
screen_width = 1920  # Zmień na rozdzielczość Twojego ekranu
screen_height = 1080  # Zmień na rozdzielczość Twojego ekranu

# Oblicz współczynnik skalowania, aby wideo pasowało do ekranu
scale_factor_x = screen_width / original_width
scale_factor_y = screen_height / original_height

# Ustaw obsługę myszki
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_rectangle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Przeskaluj klatkę do rozdzielczości ekranu
    frame_resized = cv2.resize(frame, (screen_width, screen_height))

    cv2.imshow('Video', frame_resized)

    # Przerwij po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnij zasoby
cap.release()
cv2.destroyAllWindows()

# Wyświetl współrzędne zaznaczonych obszarów
print("Zaznaczone obszary (x, y, szerokość, wysokość w oryginalnej rozdzielczości):")
for rect in rectangles:
    print(rect)
