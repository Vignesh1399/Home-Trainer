from tkinter import *
from tkinter import filedialog

import camera
import numpy as np
import math
import glob
import os

import utils
from parse import load_ps
from pprint import pprint
from scipy.signal import medfilt

# helper funtions

selected_exercise = 'none'

supported_exercises = ['bicep curl', 'squat']


def load_features_bicep(names, train=True):
    output1 = []  # List of upper arm torso angles
    output2 = []  # List of forearm upper arm angles
    for filename in names:
        ps = 0
        if (train):
            ps = load_ps('poses_compressed/bicep_curl/' + filename)
        else:
            ps = load_ps(filename)
        poses = ps.poses

        right_present = [
            1 for pose in poses if pose.rshoulder.exists and pose.relbow.exists
            and pose.rwrist.exists
        ]
        left_present = [
            1 for pose in poses if pose.lshoulder.exists and pose.lelbow.exists
            and pose.lwrist.exists
        ]
        right_count = sum(right_present)
        left_count = sum(left_present)
        side = 'right' if right_count > left_count else 'left'

        if side == 'right':
            joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip,
                       pose.neck) for pose in poses]
        else:
            joints = [(pose.lshoulder, pose.lelbow, pose.lwrist, pose.lhip,
                       pose.neck) for pose in poses]

        # filter out data points where a part does not exist
        joints = [
            joint for joint in joints if all(part.exists for part in joint)
        ]

        upper_arm_vecs = np.array([(joint[0].x - joint[1].x,
                                    joint[0].y - joint[1].y)
                                   for joint in joints])
        torso_vecs = np.array([(joint[4].x - joint[3].x,
                                joint[4].y - joint[3].y) for joint in joints])
        forearm_vecs = np.array([(joint[2].x - joint[1].x,
                                  joint[2].y - joint[1].y)
                                 for joint in joints])

        print(upper_arm_vecs.shape)
        upper_arm_vecs = upper_arm_vecs / np.expand_dims(
            np.linalg.norm(upper_arm_vecs, axis=1), axis=1)
        torso_vecs = torso_vecs / np.expand_dims(
            np.linalg.norm(torso_vecs, axis=1), axis=1)
        forearm_vecs = forearm_vecs / np.expand_dims(
            np.linalg.norm(forearm_vecs, axis=1), axis=1)

        upper_arm_torso_angle = np.degrees(
            np.arccos(
                np.clip(
                    np.sum(np.multiply(upper_arm_vecs, torso_vecs), axis=1),
                    -1.0, 1.0)))
        upper_arm_torso_angle_filtered = medfilt(
            medfilt(upper_arm_torso_angle, 5), 5)

        upper_arm_forearm_angle = np.degrees(
            np.arccos(
                np.clip(
                    np.sum(np.multiply(upper_arm_vecs, forearm_vecs), axis=1),
                    -1.0, 1.0)))
        upper_arm_forearm_angle_filtered = medfilt(
            medfilt(upper_arm_forearm_angle, 5), 5)

        output1.append(upper_arm_torso_angle_filtered.tolist())
        output2.append(upper_arm_forearm_angle_filtered.tolist())
    return output1, output2


def load_features_squat(names, train=True):
    output1 = []  # List of upper leg torso angles
    output2 = []  # List of upper leg lower leg angles
    # find the arm that is seen most consistently
    for filename in names:
        ps = 0
        if (train):
            ps = load_ps('poses_compressed/squat/' + filename)
        else:
            ps = load_ps(filename)
        poses = ps.poses
        #print(poses)
        right_present = [
            1 for pose in poses
            if pose.rhip.exists and pose.rknee.exists and pose.rankle.exists
        ]
        left_present = [
            1 for pose in poses
            if pose.lhip.exists and pose.lknee.exists and pose.lankle.exists
        ]
        right_count = sum(right_present)
        left_count = sum(left_present)
        side = 'right' if right_count > left_count else 'left'

        print('Exercise leg detected as: {}.'.format(side))

        if side == 'right':
            joints = [(pose.rhip, pose.rknee, pose.rankle, pose.neck)
                      for pose in poses]
        else:
            joints = [(pose.lhip, pose.lknee, pose.lankle, pose.neck)
                      for pose in poses]
        #print(joints)

        # filter out data points where a part does not exist
        joints = [
            joint for joint in joints if all(part.exists for part in joint)
        ]

        upper_leg_vecs = np.array([(joint[0].x - joint[1].x,
                                    joint[0].y - joint[1].y)
                                   for joint in joints])
        torso_vecs = np.array([(joint[3].x - joint[0].x,
                                joint[3].y - joint[0].y) for joint in joints])
        lower_leg_vecs = np.array([(joint[2].x - joint[1].x,
                                    joint[2].y - joint[1].y)
                                   for joint in joints])

        # normalize vectors
        upper_leg_vecs = upper_leg_vecs / np.expand_dims(
            np.linalg.norm(upper_leg_vecs, axis=1), axis=1)
        torso_vecs = torso_vecs / np.expand_dims(
            np.linalg.norm(torso_vecs, axis=1), axis=1)
        lower_leg_vecs = lower_leg_vecs / np.expand_dims(
            np.linalg.norm(lower_leg_vecs, axis=1), axis=1)

        # filtering
        upper_leg_torso_angles = np.degrees(
            np.arccos(
                np.clip(
                    np.sum(np.multiply(upper_leg_vecs, torso_vecs), axis=1),
                    -1.0, 1.0)))
        upper_leg_torso_angles_filtered = medfilt(
            medfilt(upper_leg_torso_angles, 5), 5)

        upper_leg_lower_leg_angles = np.degrees(
            np.arccos(
                np.clip(
                    np.sum(np.multiply(upper_leg_vecs, lower_leg_vecs),
                           axis=1), -1.0, 1.0)))
        upper_leg_lower_leg_angles_filtered = medfilt(
            medfilt(upper_leg_lower_leg_angles, 5), 5)

        output1.append(upper_leg_torso_angles_filtered.tolist())
        output2.append(upper_leg_lower_leg_angles_filtered.tolist())

    return output1, output2


def get_feedback_bicep(npy_path):
    files = utils.files_in_order('poses_compressed/bicep_curl')
    y_train = utils.get_labels(files)
    X_train_1, X_train_2 = load_features_bicep(files)
    X_test_1, X_test_2 = load_features_bicep([npy_path], train=False)
    predictions = []
    feedback = []
    for example in range(1):
        # Store the average distance to good and bad training examples
        f1_good, f1_bad, f2_good, f2_bad = [[] for i in range(4)]

        # Compare distance of current test example with all training examples
        for i in range(len(X_train_1)):
            dist1 = utils.DTWDistance(X_train_1[i], X_test_1[example])
            dist2 = utils.DTWDistance(X_train_2[i], X_test_2[example])
            if y_train[i]:
                f1_good.append(dist1)
                f2_good.append(dist2)
            else:
                f1_bad.append(dist1)
                f2_bad.append(dist2)
        good_score = np.mean(f1_good) + np.mean(f2_good)
        bad_score = np.mean(f1_bad) + np.mean(f2_bad)
        if good_score < bad_score:
            predictions.append(1)
            feedback.append("Exercise was performed correctly, Cheers!!!")
        else:
            predictions.append(0)
            feedback_string = ""
            if (np.mean(f1_good) > np.mean(f1_bad)):
                feedback_string += "\n\u2022 Your upper arm shows significant rotation around the shoulder when curling. Try holding your upper arm still, parallel to your chest, and concentrate on rotating around your elbow only.\n"

            if (np.mean(f2_good) > np.mean(f2_bad)):
                feedback_string += "\n\u2022 You are not curling the weight all the way to the top, up to your shoulders. Try to curl your arm completely so that your forearm is parallel with your torso. It may help to use lighter weight.\n"
            feedback.append(feedback_string)

    return predictions[0], feedback[0]


def get_feedback_squat(npy_path):
    files = utils.files_in_order('poses_compressed/squat')
    y_train = utils.get_labels(files)
    X_train_1, X_train_2 = load_features_squat(files)
    X_test_1, X_test_2 = load_features_squat([npy_path], train=False)
    predictions = []
    feedback = []
    for example in range(1):
        # Store the average distance to good and bad training examples
        f1_good, f1_bad, f2_good, f2_bad = [[] for i in range(4)]

        # Compare distance of current test example with all training examples
        for i in range(len(X_train_1)):
            dist1 = utils.DTWDistance(X_train_1[i], X_test_1[example])
            dist2 = utils.DTWDistance(X_train_2[i], X_test_2[example])
            if y_train[i]:
                f1_good.append(dist1)
                f2_good.append(dist2)
            else:
                f1_bad.append(dist1)
                f2_bad.append(dist2)
        good_score = np.mean(f1_good) + np.mean(f2_good)
        bad_score = np.mean(f1_bad) + np.mean(f2_bad)
        if good_score < bad_score:
            predictions.append(1)
            feedback.append("Exercise was performed correctly, Cheers!!!")
        else:
            predictions.append(0)
            feedback_string = ""
            if (np.mean(f1_good) > np.mean(f1_bad)):
                feedback_string += "\n\u2022 You are bending your back, Make sure your thighs are perpendicular to your back\n"
            if (np.mean(f2_good) > np.mean(f2_bad)):
                feedback_string += "\n\u2022 You are not bending your knees enough, Make sure the knees reach a position parallel to the ground\n"
            feedback.append(feedback_string)

    return predictions[0], feedback[0]


def send_command():
    global selected_exercise
    m_message = messageWindow.get('1.0', END)
    if m_message == '\n' or m_message == ' \n':
        messageWindow.delete('1.0', END)
        return
    chatWindow.insert(END, "\n" + m_message)

    # compare message and do needy stuff
    if m_message[:-1] in supported_exercises:
        selected_exercise = m_message[:-1]
        chatWindow.insert(END, "\n" + "You Selected " + selected_exercise)
        chatWindow.insert(
            END, "\n\n" +
            "Where do you want to select the input from\n(camera | file | npyfile)"
        )

    if m_message == 'file\n':
        if selected_exercise == 'none':
            chatWindow.insert(END, "\n" + "Please select exercise first")
        else:
            root.filename = filedialog.askopenfilename(
                initialdir=".",
                title="Exercise Video Clip",
                filetypes=(("all files", "*"), ("all files", "*.*")))
            video_path = root.filename
            chatWindow.insert(END, "\n" + video_path)
            # video = os.path.basename(args.video)

            current_dir = os.getcwd()
            
            output_path = os.path.join(current_dir, 'poses')
            os.chdir('openpose')
            openpose_path = os.path.join('bin', 'OpenPoseDemo.exe')
            subprocess.call([openpose_path, 
                            '--video', os.path.join('..', video_path), 
                            '--write_json', output_path])
            parse_sequence(output_path, current_dir)
            pose_seq = load_ps(os.path.join(current_dir, 'keypoints.npy'))
            os.remove(os.path.join(current_dir, 'keypoints.npy'))
            #print('\n\n\n\n\n\n')
            #print('pose_seq' ,pose_seq)
            #print('\n\n\n\n\n\n')
            (correct, feedback) = evaluate_pose(pose_seq, selected_exercise)
            if correct:
                chatWindow.insert(END, "Exercise performed correctly!\n")
                print('Exercise performed correctly!')
            else:
                chatWindow.insert(END, "Exercise could be improved: \n")
                print('Exercise could be improved:')
            chatWindow.insert(END, feedback)

    if m_message == 'npyfile\n':
        if selected_exercise == 'none':
            chatWindow.insert(END, "\n" + "Please select exercise first")
        else:
            root.filename = filedialog.askopenfilename(
                initialdir=".",
                title="Exercise npy file",
                filetypes=(("all files", "*"), ("all files", "*.*")))
            npy_path = root.filename
            chatWindow.insert(END, "\nFile Path: " + npy_path + "\n")
            # video = os.path.basename(args.video)
            feedback = 0
            if (selected_exercise == 'squat'):
                prediction, feedback = get_feedback_squat(npy_path)
            else:
                prediction, feedback = get_feedback_bicep(npy_path)
            chatWindow.insert(END, feedback + "\n")

    if m_message == 'camera\n':
        if selected_exercise == 'none':
            chatWindow.insert(END, "\n" + "Please select exercise first")
        else:
            camera.record()
            chatWindow.insert(END, "\n" + camera.file_path)

    messageWindow.delete('1.0', END)


# Root Widget

root = Tk()

root.title('HomeTrainer Chat Bot')
root.geometry('400x478')
root.resizable(width=True, height=True)

main_menu = Menu(root)

# Create the submenu

file_menu = Menu(root)
help_menu = Menu(root)

# Add commands to submenu

file_menu.add_command(label='Exit', command=root.destroy)
help_menu.add_command(label='About')

# Add file menu to main menu

main_menu.add_cascade(label='File', menu=file_menu)
main_menu.add_cascade(label='Help', menu=help_menu)
root.config(menu=main_menu)

# Create a window for the conversation and place it on the parent window

chatWindow = Text(
    root,
    bd=1,
    bg='black',
    width='50',
    height='8',
    font=('Arial', 10),
    foreground='#00ffff',
)
chatWindow.place(x=6, y=6, height=385, width=370)

# Create a button to send the message and place it on the parent window.

Button = Button(root,
                text='Send',
                width='8',
                height=5,
                bd=0,
                bg='#0080ff',
                activebackground='#00bfff',
                foreground='#ffffff',
                font=('Arial', 12),
                command=send_command)
Button.place(x=6, y=400, height=60)

# Create the text area where the messages will be entered and place it on the parent window.

messageWindow = Text(
    root,
    bd=0,
    bg='black',
    width='30',
    height='4',
    font=('Arial', 10),
    foreground='#00ffff',
)
messageWindow.place(x=128, y=400, height=60, width=260)

# Create a scroll bar and place it on the parent window.

scrollbar = Scrollbar(root, command=chatWindow.yview, cursor='star')
scrollbar.place(x=375, y=5, height=385)

# Run the main loop.

chatWindow.insert(END, "---------- ❚█══█❚ ----------")
chatWindow.insert(END, "\n\n           HomeTrainer\n")
chatWindow.insert(END, "\n---------- ❚█══█❚ ----------")
chatWindow.insert(
    END, "\n\nWhat Exercise do you want to perform? squat | bicep curl\n")

root.mainloop()
