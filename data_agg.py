import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import io
import pyautogui


class DataAgg:
	def __init__(self,):
		# List to store past emotions
		self.emotion_mem = []
		self.video_capture = cv2.VideoCapture(0)
		self.start_ml()
		self.events = [] # list of tuples of times and pictures

	def start_ml(self,):
		pass # init ml stuff here

	def run_ml_on_img(self, img):
		# emo = self.model # Query model here

		# THIS CODE JUST RANDOMIZES EMOTIONS
		emo = np.random.uniform(size=(7,))
		emo = emo - 0.5
		emo = emo * np.array([3, 2, 2, 2, 2, 2, 2])
		print(emo)
		if len(self.emotion_mem) > 0:
			emo = emo + self.emotion_mem[-1]

		# KEEP THIS CODE IT FLOORS/CEILS EVERYTHING
		print(emo)
		emo = np.round(emo).astype(int)
		emo = np.minimum(emo, 3)
		emo = np.maximum(emo, [-3, 0, 0, 0, 0, 0, 0])
		print(emo)
		self.emotion_mem.append(emo)

		# This is perminent event detection stuff
		if len(self.emotion_mem) > 60 and len(self.emotion_mem) % 15 == 0:
			current_avg = np.stack([emos for emos in self.emotion_mem[-30:]])
			current_avg = np.mean(current_avg, axis=0)

			previous_avg = np.stack([emos for emos in self.emotion_mem[-60:-30]])
			previous_avg = np.mean(previous_avg, axis=0)

			# Capture event if any emotion goes up signficantly or sentiment goes down
			if np.any(np.abs(current_avg[1:] - previous_avg[1:]) > 1) or np.abs(current_avg[0] - previous_avg[0]) > 1.5:
				screenshot = pyautogui.screenshot().resize((512, 288))
				screenshot = ImageTk.PhotoImage(image=screenshot)
				self.events.append((len(self.emotion_mem) - 1, screenshot))


	def request_new_img(self,):
		# _, frame = self.video_capture.read()
		frame = cv2.imread("assets/test_img.jpg")

		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		self.run_ml_on_img(img)

		img = Image.fromarray(img).resize((512, 288))
		img = ImageTk.PhotoImage(image=img)

		return img

	def get_event_img(self, i):
		if len(self.events) > i:
			return self.events[i][1] # stuff stored in self.events must be already ImageTk
		else:
			img = Image.fromarray(np.zeros((512, 288, 3)).astype(np.uint8)).resize((512, 288))
			return ImageTk.PhotoImage(image=img)

	def get_event_emo(self, i, emo_i):
		if len(self.events) > i:
			time = self.events[i][0]
			return self.emotion_mem[time][emo_i]
		else:
			return 0

	def get_emotion(self, i):
		try:
			return self.emotion_mem[-1][i]
		except IndexError:
			return 0

	def get_graph(self, live=True, cur_event=0):
		print("graph")
		plt.close()
		fig, ax1 = plt.subplots(figsize=(7, 2.6))

		if live:
			sent_data = np.convolve(np.array(self.emotion_mem)[-240:, 0], np.ones((10,)), mode="same")/10
			ax1.plot(0 - np.array(range(len(sent_data)))[::-1], sent_data, color="tab:blue", linewidth=3)
		else:
			sent_data = np.convolve(np.array(self.emotion_mem)[:, 0], np.ones((10,)), mode="same")/10
			ax1.plot(range(len(sent_data)), sent_data, color="tab:blue", linewidth=3)

		ax1.set_xlabel("Time (s)")
		ax1.set_ylabel("Overall", color="tab:blue")
		ax1.tick_params(axis='y', labelcolor="tab:blue")
		ax1.set_ylim(-3, 3)

		ax2 = ax1.twinx()
		ax2.set_ylabel("Individual Emotions", color="black")
		ax2.set_ylim(0, 3)

		for i in range(6):
			label = ["Happiness", "Sadness", "Anger", "Fear", "Disgust", "Suprise"][i]
			color = ["tab:green", "tab:olive", "tab:orange", "tab:purple", "tab:brown", "tab:pink"][i]
			if live:
				sent_data = np.convolve(np.array(self.emotion_mem)[-240:, i + 1], np.ones((10,)), mode="same")/10
				ax2.plot(0 - np.array(range(len(sent_data)))[::-1], sent_data, label=label, color=color)
			else:
				sent_data = np.convolve(np.array(self.emotion_mem)[:, i + 1], np.ones((10,)), mode="same")/10
				ax2.plot(range(len(sent_data)), sent_data, label=label, color=color)
		ax2.legend(loc="upper left")

		if not live and len(self.events) > 0:
			for i, event in enumerate(self.events):
				if i == cur_event:
					ax1.vlines(event[0], -3, 3, color="tab:red", linewidth=5)
				else:
					ax1.vlines(event[0], -3, 3, color="black", linewidth=5)

		plt.tight_layout()

		print("graph 2")
		buf = io.BytesIO()
		fig.savefig(buf, format='png')
		buf.seek(0)
		img = Image.open(buf).resize((700, 260))
		img = ImageTk.PhotoImage(image=img)
		buf.close()
		return img







