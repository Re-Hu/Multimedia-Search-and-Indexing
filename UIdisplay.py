import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from PIL import Image, ImageTk

class VideoPlayer:
    def __init__(self, frame_i, output_video_file, video_file, audio_file):
        self.video_file = video_file
        self.output_video_file = output_video_file
        self.audio_file = audio_file
        self.width = 352
        self.height = 288

        self.is_playing = True
        self.initial = True
        self.frame_i = frame_i
        self.start_time = self.frame_i / 30
        self.pause = False

        # self.convert()
        # print("output_video_file: ", self.output_video_file, "audio_file: ", self.audio_file, "video_file: ",self.video_file)
        self.setup_gui()
        self.setup_video()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Video Player")

        video_frame = tk.Frame(self.root)
        video_frame.pack()

        self.canvas = tk.Canvas(video_frame, width=self.width, height=self.height)
        self.canvas.pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack()

        play_button = ttk.Button(button_frame, text="Play", command=self.play_video)
        play_button.pack(side=tk.LEFT, padx=10)

        pause_button = ttk.Button(button_frame, text="Pause", command=self.pause_video)
        pause_button.pack(side=tk.LEFT, padx=10)

        reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_video)
        reset_button.pack(side=tk.LEFT, padx=10)

    def setup_video(self):
        pygame.mixer.init()
        pygame.mixer.music.load(self.audio_file)

        self.cap = cv2.VideoCapture(self.video_file)

    def convert(self):
        with open(self.video_file, 'rb') as f:
            video = f.read()

        num_frames = int(len(video) / (self.width * self.height * 3))
        frame_image = np.frombuffer(video, dtype=np.uint8).reshape((num_frames, self.height, self.width, 3))

        # output_video_file = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_file, fourcc, 30, (self.width, self.height))

        for frame in frame_image:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

    def play_video(self):
        self.is_playing = True
        if self.pause:
            self.pause = False
            pygame.mixer.music.unpause()
        else:
            pygame.mixer.music.play(start=self.start_time)

    def pause_video(self):
        self.pause = True
        self.is_playing = False
        pygame.mixer.music.pause()

    def reset_video(self):
        self.is_playing = False
        self.start_time = 0
        pygame.mixer.music.rewind()
        pygame.mixer.music.pause()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=frame_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.photo = photo


    def update_frame(self):
        if self.is_playing:
            if self.initial:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_i)
                self.is_playing = False
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=frame_image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.photo = photo
        self.root.after(30, self.update_frame)

    def run(self):
        self.update_frame()
        self.initial = False
        self.root.mainloop()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

# video_file = 'Queries/RGB_Files/video4_1.rgb'
# output_video_file = 'output_video.mp4'
# audio_file = "Queries/Audios/video4_1.wav"
# frame_i = 200
#
# video_player = VideoPlayer(frame_i, output_video_file, video_file, audio_file)
# video_player.run()
