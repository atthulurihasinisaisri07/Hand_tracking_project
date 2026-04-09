import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]
overlayImages = []
finger_history = []

for i in range(1, 6):
    img = cv2.imread(f"images/{i}.png")
    overlayImages.append(img)



while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

        fingers = []

        if lmList:
            # Thumb
            if lmList[4][1] > lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other 4 fingers
            for i in range(1, 5):
                if lmList[tipIds[i]][2] < lmList[tipIds[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            if lmList:

                fingerCount = fingers.count(1)
                # ---- Smooth finger count ----
                finger_history.append(fingerCount)

            if len(finger_history) > 5:
                finger_history.pop(0)

                fingerCount = max(set(finger_history), key=finger_history.count)


            if fingerCount > 0 and fingerCount <= 5:
                overlay = overlayImages[fingerCount - 1]

                # Resize overlay image
                overlay = cv2.resize(overlay, (200, 200))

                h, w, _ = overlay.shape
                img[0:h, img.shape[1]-w:img.shape[1]] = overlay


    cv2.imshow("Hand Tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
