try:
    from cv2 import cv2
except ImportError:
    pass

faceCascade = cv2.CascadeClassifier('./haarFeatures/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('./haarFeatures/haarcascade_smile.xml')


def detectSmile(grayScaleImage, colorImage):
    # First task is to detect face
    face = faceCascade.detectMultiScale(
        image=grayScaleImage,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (topFaceX, topFaceY, widthFace, heightFace) in face:
        # Draw a red rectangle over the face
        cv2.rectangle(
            img=colorImage,
            pt1=(topFaceX, topFaceY),
            pt2=((topFaceX + widthFace), (topFaceY + heightFace)),
            thickness=1,
            color=(0, 0, 255)
        )
        faceGrayScale = grayScaleImage[topFaceY: topFaceY + heightFace, topFaceX:topFaceX + widthFace]
        faceColored = colorImage[topFaceY: topFaceY + heightFace, topFaceX:topFaceX + widthFace]

        # Find the smiley face within the face
        smile = smileCascade.detectMultiScale(
            image=faceGrayScale,
            scaleFactor=1.7,
            minNeighbors=22
        )

        for (_, _, _, _) in smile:
            # Draw a green rectangle over the face that smiles
            cv2.rectangle(
                img=colorImage,
                pt1=(topFaceX, topFaceY),
                pt2=((topFaceX + widthFace), (topFaceY + heightFace)),
                thickness=2,
                color=(0, 255, 0)
            )

    return colorImage


capture = cv2.VideoCapture(0)
while True:
    _, frame = capture.read()
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = detectSmile(grayScale, frame)
    cv2.imshow('Video', frame)
    if cv2.waitKey(20) and 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
