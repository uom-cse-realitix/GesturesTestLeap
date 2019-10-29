import ctypes
import sys
import cv2
import numpy as np
from sklearn.externals import joblib


import Leap

svm_model = joblib.load("./models/svm_v3.pkl")
command_classes = ['Pointing', 'Capture', 'ZoomIn', 'ZoomOut', 'Roaming']

def run(controller):
    capture_gesture = False
    gesture_sequence = np.array([])
    gesture = ''
    while True:
        frame = controller.frame()
        image = frame.images[1]
        if image.is_valid:
            i_address = int(image.data_pointer)
            ctype_array_def = ctypes.c_ubyte * image.height * image.width
            as_ctype_array = ctype_array_def.from_address(i_address)
            as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
            img = np.reshape(as_numpy_array, (image.height, image.width))
            if capture_gesture:
                for hand in frame.hands:
                    pv = []
                    av = []
                    prev_finger = None
                    c = hand.palm_position
                    m = 1
                    for finger in hand.fingers:
                        if prev_finger:
                            ad = finger.tip_position.distance_to(prev_finger.tip_position)
                            av.append(ad)
                            prev_finger = finger
                        else:
                            prev_finger = finger
                        pd = finger.tip_position.distance_to(c)
                        m = pd if finger.type == Leap.Finger.TYPE_MIDDLE else m
                        pv.append(pd)
                    gesture_sequence = np.append(gesture_sequence, np.array(pv) / m)
                    gesture_sequence = np.append(gesture_sequence, np.array(av) / m)

                    img = cv2.putText(img, '%s' % (gesture), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if len(gesture_sequence) > 270:
                        prediction = svm_model.predict([gesture_sequence[:270]])
                        gesture = command_classes[int(prediction[0])]

                        print(gesture)
                        gesture_sequence = gesture_sequence[90:]

            cv2.imshow('Image', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                capture_gesture = not capture_gesture
                print("Capturing gestures %s.." % ("finished", "started")[capture_gesture])


def main():
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
    try:
        run(controller)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == '__main__':
    main()
