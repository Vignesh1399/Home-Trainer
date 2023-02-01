#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import the library

from tkinter import *
from tkinter import filedialog

import camera
from mega import parse_sequence, load_ps, evaluate_pose
import os
import subprocess

# helper funtions

selected_exercise = 'none'

supported_exercises = ['bicep curl', 'squat']

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
        chatWindow.insert(END, "\n\n" + "Where do you want to select the input from\n(camera | file | npyfile)")
    
    if m_message == 'file\n':
        if selected_exercise == 'none':
            chatWindow.insert(END, "\n" + "Please select exercise first")
        else:
            root.filename =  filedialog.askopenfilename(
                initialdir = ".",
                title = "Exercise Video Clip",
                filetypes = (("all files","*"),
                ("all files","*.*"))
            )
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
                chatWindow.insert(END, "\nExercise performed correctly!\n")
                print('Exercise performed correctly!')
            else:
                chatWindow.insert(END, "\nExercise could be improved: \n")
                print('Exercise could be improved:')
            chatWindow.insert(END, feedback)
        
    if m_message == 'npyfile\n':
        if selected_exercise == 'none':
            chatWindow.insert(END, "\n" + "Please select exercise first")
        else:
            root.filename =  filedialog.askopenfilename(
                initialdir = ".",
                title = "Exercise npy file",
                filetypes = (("all files","*"),
                ("all files","*.*"))
            )
            npy_path = root.filename
            chatWindow.insert(END, "\nFile Path: " + npy_path + "\n")
            # video = os.path.basename(args.video)
            
            pose_seq = load_ps(npy_path)
            (correct, feedback) = evaluate_pose(pose_seq, selected_exercise)
            if correct:
                # chatWindow.insert(END, "Exercise performed correctly!\n")
                print('\nExercise performed correctly!')
            else:
                chatWindow.insert(END, "\nExercise could be improved: \n")
                print('Exercise could be improved:')
            chatWindow.insert(END, feedback)
            
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

Button = Button(
    root,
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
chatWindow.insert(END, "\n\nWhat Exercise do you want to perform? squat | bicep curl\n")


root.mainloop()
