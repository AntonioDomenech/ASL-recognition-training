import tensorflow as tf
import numpy as np
import random
import os

# txtpath = 'path'
# signs = ['hello', 'sign', 'no', 'understand'] #... add the signs you have ...
# handlandmarks = []
# signlandmarks = []
# training_data = []
# i = 0
# #for each folder we take the path and set an index for the sign
# for sign in signs:
#     #try:
#     path = os.path.join(txtpath, sign)
#     sign_num = signs.index(sign)
#     print(sign)
#     #for each txt file we open it, read it and create the training data
#     for txt in os.listdir(path):
#         #try:
#         #print(txt)
#         with open(path+'/'+txt, 'r') as reader:
#             #for each line in the txt file (there is only one)
#             for line in reader:
#                 #for each number in the line we build the list of points for the landmarks
#                 for number in line.split():
#                     if i < 84:
#                         handlandmarks.append(round(float(number),3))
#                         i = i + 1
#                     else:
#                         signlandmarks.append(handlandmarks)
#                         handlandmarks = []
#                         i = 0
#         signlandmarks = np.asarray(signlandmarks)
#         if signlandmarks.shape[0] >= 20:
#             signlandmarks = signlandmarks[0:20]
#             training_data.append([signlandmarks, sign_num])
#             #print(signlandmarks.shape)
#         signlandmarks = []
#         #except:
#          #   pass
#     #except:
#      #   pass
# print(len(training_data))

# print('FINISHED')

model = tf.keras.models.load_model('/home/andope16/Documents/AI_test_1/hsnu.model')

# direction = 'cd /home/andope16/Documents/MediaPipe200806/mediapipe/'
# # os.system(direction)

# setting = 'bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11   mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_gpu'
# # os.system(setting)

# os.system(direction)

# cmd = 'GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_gpu   --calculator_graph_config_file=mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt --input_video_path='+'/home/andope16/Videos/newSign/original/hello/helloADL2.mp4'+' --output_video_path='+'/home/andope16/Videos/newSign/original/hello/test1.mp4'

# os.system(cmd)
try:
    pred1 = 100
    while(True):
        #try:
            tryal = []
            past_line = []
            frames = []
            for i in range(20):
                frame_landmarks = []
                for j in range(2):
                    val = True
                    while val == True:
                        with open('landmark_test.txt', 'r') as reader:
                            for line in reader:
                                tryal.append(line)
                                if past_line == tryal:
                                    tryal = []
                                else:
                                    past_line = tryal
                                    tryal = []
                                    for number in line.split():
                                        frame_landmarks.append(round(float(number), 3))
                                    val = False
                #frame_landmarks = np.asarray(frame_landmarks)
                #print(frame_landmarks.shape)
                frames.append(frame_landmarks)
            frames = np.asarray(frames)
            #print(frames[0].shape)
            # print(frames[0])
            # print(frames[1])
            # print(frames[2])
            # print(frames[3])
            # print(frames[4])
            # break
            #pack.append(frames)
            #print(frames[0])

            # sign = []
            # for i in range(20):
            #     rand = np.random.random(84)
            #     sign.append(rand)
            # print(sign)

            sign = np.array(frames).reshape(-1, 20, 84)

            # categories = ['Hello', 'No', 'Understand', 'Sign']

            prediction = model.predict([sign])
            pred = np.argmax(prediction)
            #print(prediction[0][pred])
            if prediction[0][pred] > 0.4:
                if pred == pred1:
                    pass
                else:
                    if pred == 0:
                        print('Hello')
                        pred1 = pred
                    if pred == 1:
                        print('Sign')
                        pred1 = pred
                    if pred == 2:
                        print('No')
                        pred1 = pred
                    if pred == 3:
                        print('Understand')
                        pred1 = pred
            else:
                pass
        #except:
            #pass
except KeyboardInterrupt:
    pass
