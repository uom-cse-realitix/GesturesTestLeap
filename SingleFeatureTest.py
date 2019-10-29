import time
import Leap
import ctypes
import cv2
import numpy as np
import sys
from multiprocessing import Process
from tensorflow import keras
from sklearn.externals import joblib

# svm_model = joblib.load("svm_v2.pkl")
model = keras.models.load_model("./models/gesture_lstm_v6.h5")
command_classes = ['Pointing', 'Capture', 'ZoomIn', 'ZoomOut', 'Roaming']

# def process_gesture(sequence):
#     from tensorflow import keras
#     model = keras.models.load_model("gesture_lstm.h5")
#     command_classes = ['Pointing', 'Capture', 'ZoomIn', 'ZoomOut', 'Roaming']
#     prediction = model.predict(sequence.reshape(1, 1, 150))
#     gesture = command_classes[np.argmax(prediction)]
#     print(gesture)


def run(controller):
    capture_gesture = False
    gesture_sequence = np.array([])
    gesture_sequence_dist = np.array([])
    gesture = ''
    while True:
        frame = controller.frame()
        image = frame.images[1]
        if image.is_valid:
            i_address = int(image.data_pointer)
            ctype_array_def = ctypes.c_ubyte * image.height * image.width
            # as ctypes array
            as_ctype_array = ctype_array_def.from_address(i_address)
            # as numpy array
            as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
            img = np.reshape(as_numpy_array, (image.height, image.width))
            if capture_gesture:
                for hand in frame.hands:
                    # print(len(gesture_sequence))
                    v = []
                    # dv = []
                    # prev_finger = None
                    c = hand.palm_position
                    m = 1
                    for finger in hand.fingers:
                        # if prev_finger:
                        #     d2 = finger.tip_position.distance_to(prev_finger.tip_position)
                        #     dv.append(d2)
                        #     prev_finger = finger
                        # else:
                        #     prev_finger = finger
                        d = finger.tip_position.distance_to(c)
                        m = d if finger.type == Leap.Finger.TYPE_MIDDLE else m
                        v.append(d)
                    gesture_sequence = np.append(gesture_sequence, np.array(v) / m)
                    # gesture_sequence_dist = np.append(gesture_sequence_dist, np.array(dv) / m)
                    img = cv2.putText(img, '%s' % (gesture), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # print(len(gesture_sequence))
                    if len(gesture_sequence) > 150:
                        '''For LSTM models'''
                        prediction = model.predict(gesture_sequence[:150].reshape(1, 1, 150))
                        gesture = command_classes[np.argmax(prediction)]
                        # temp = np.greater(prediction, 0.5)[0]
                        # gesture = command_classes[4 if len(temp) == 0 else temp[0]]
                        '''For SVM classification'''
                        # prediction = svm_model.predict([gesture_sequence[:150]])
                        # gesture = command_classes[int(prediction[0])]

                        print(gesture)
                        # p = Process(target=process_gesture, args=(gesture_sequence[:150],))
                        gesture_sequence = gesture_sequence[50:]
                        # p.start()

            cv2.imshow('Image', img)
            # time.sleep(0.01)
            # print("Timestamp: %d, hands: %d, fingers: %d" % (frame.timestamp, len(frame.hands), len(frame.fingers)))
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

#
# gesture_sequence = np.array([])
#
# for i in range(1000):
#     seq = np.random.rand(5)
#     gesture_sequence = np.append(gesture_sequence, seq)
#     if len(gesture_sequence) > 150:
#         p = Process(target=process_gesture, args=(gesture_sequence[:150],))
#         gesture_sequence = gesture_sequence[100:]
#         p.start()
#     time.sleep(0.2)
