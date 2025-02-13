import cv2

drawing = False  # True, if mouse button is pressed
ix, iy = -1, -1  # Initial rectangle coordinates
rectangles = []  # Lista to store all selected areas
scale_factor_x = 1
scale_factor_y = 1

# Mouse handling functions
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, frame

    # Scales mouse coordinates to the original resolution
    real_x = int(x / scale_factor_x)
    real_y = int(y / scale_factor_y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = real_x, real_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Create a temporary rectangle on current frame
            frame_copy = frame.copy()
            # Scales coordinates back to the original resolution
            cv2.rectangle(frame_copy, (int(ix * scale_factor_x), int(iy * scale_factor_y)),
                          (x, y), (0, 255, 0), 2)
            cv2.imshow('Video', frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Drawing final rectangle on frame
        cv2.rectangle(frame, (int(ix * scale_factor_x), int(iy * scale_factor_y)),
                      (x, y), (0, 255, 0), 2)
        rectangles.append((ix, iy, real_x - ix, real_y - iy))  # Store (x, y, width, height)
        cv2.imshow('Video', frame)
        print(f"Selected area: {(ix, iy, real_x - ix, real_y - iy)}")

# Read video
video_path = '2024-09-20 15-21-39.mkv'
cap = cv2.VideoCapture(video_path)

# Original video size
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Screen resolution
screen_width = 1920  # Change to your screen resolution
screen_height = 1080  # Change to your screen resolution

# Calculate scale factor so it fits the screen
scale_factor_x = screen_width / original_width
scale_factor_y = screen_height / original_height

# Set mouse handling
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_rectangle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Scale frame to the screen resolution
    frame_resized = cv2.resize(frame, (screen_width, screen_height))

    cv2.imshow('Video', frame_resized)

    # Quit after pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Show selected areas coordinates
print("Selected areas: (x, y, width, height in original resolution):")
for rect in rectangles:
    print(rect)
