import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import io
import pyautogui
from deepface import DeepFace


# Hardcoded settings
SENSITIVITY = 5


class DataAgg:
	def __init__(self,):
		# List to store past emotions
		self.emotion_mem = []
		self.video_capture = cv2.VideoCapture(0)
		self.start_ml()
		self.events = [] # list of tuples of times and pictures
		self.alert_mode = False

	def start_ml(self,):
		DeepFace.analyze("assets/test_img.jpg", actions=["emotion"], detector_backend="ssd")

	def run_ml_on_img(self, img):
		try:
			self.alert_mode = False
			# use deepface analyze function to detect emotion
			# returns list of dicts including emotion scores, detected face region, dominant emotion, and confidence of face detection
			emo = DeepFace.analyze(img, actions=["emotion"], detector_backend="ssd")  # img must be np array in BGR format
			emo = emo[np.argmax([e["face_confidence"] for e in emo])]
			print("---------------------- NEW ITERATION ----------------------")
			print("raw emotion score distrib:", emo["emotion"])
			emo = np.array([emo["emotion"][e] for e in [
				"happy", "sad", "angry", "fear", "disgust", "surprise"
			]])
			print("----")

			# get scores for emotion and sentiment
			emo = (np.exp(emo/SENSITIVITY) - np.exp(-emo/SENSITIVITY))/(np.exp(emo/SENSITIVITY) + np.exp(-emo/SENSITIVITY))
			# Add sentiment (happiness - sadness)
			emo = np.insert(emo, 0, emo[0] - emo[1])
			emo *= 3
			print("emotion score distrib (excl. neutral):", emo)
			print("sentiment score:", emo[0])
			print("----")

		# default values of zero
		except ValueError:
			emo = np.array([0, 0, 0, 0, 0, 0, 0])

		# encode scores for emotion and sentiment (values between 0-3 emotion and -3-3 sentiment)
		emo = np.round(emo).astype(int)
		emo = np.minimum(emo, 3)
		emo = np.maximum(emo, [-3, 0, 0, 0, 0, 0, 0])
		print("emotion values:", emo)
		print("sentiment value:", emo[0])
		print("----")
		self.emotion_mem.append(emo)  # store emotion data

		# This is perminent event detection stuff
		if len(self.emotion_mem) > 60 and len(self.emotion_mem) % 15 == 0:
			current_avg = np.stack([emos for emos in self.emotion_mem[-30:]])
			current_avg = np.mean(current_avg, axis=0)

			previous_avg = np.stack([emos for emos in self.emotion_mem[-60:-30]])
			previous_avg = np.mean(previous_avg, axis=0)

			# Capture event if any emotion goes up signficantly or sentiment goes down
			emo = None
			if np.abs(current_avg[0] - previous_avg[0]) > 3: # Sentiment changed
				emo = "Overall Sentiment"
				bad = current_avg[0] < previous_avg[0]
			elif np.abs(current_avg[1] - previous_avg[1]) > 1.5: # Happiness changed
				emo = "Happiness"
				bad = current_avg[1] < previous_avg[1]
			elif np.abs(current_avg[2] - previous_avg[2]) > 1.5: # Sadness changed
				emo = "Sadness"
				bad = current_avg[2] > previous_avg[2]
			elif np.abs(current_avg[3] - previous_avg[3]) > 1.5: # Anger changed
				emo = "Anger"
				bad = current_avg[3] > previous_avg[3]
			elif np.abs(current_avg[4] - previous_avg[4]) > 1.5: # Fear changed
				emo = "Fear"
				bad = current_avg[4] > previous_avg[4]
			elif np.abs(current_avg[5] - previous_avg[5]) > 1.5: # Disgust changed
				emo = "Disgust"
				bad = current_avg[5] > previous_avg[5]
			elif np.abs(current_avg[6] - previous_avg[6]) > 1.5: # Suprise changed
				emo = "Suprise"
				bad = current_avg[5] > previous_avg[5]

			if emo:
				screenshot = pyautogui.screenshot().resize((512, 288))
				screenshot = ImageTk.PhotoImage(image=screenshot)
				self.events.append((len(self.emotion_mem) - 1, screenshot, emo, bad))
				# console output
				print("\nALERT: Significant change in emotion or sentiment! Event logged...")
				print("Detected significant shift of: %s. Worsened: %s.\n" % (emo, bad))

				# This begins the trigger of a notification when a new event is recorded
				# if emo in ["Overall sentiment", "Sadness", "Anger", "Fear", "Disgust"]:
				# 	self.alert_mode = True
	
				# right now, just testing the notification system works when events are detected in general
				self.alert_mode = True

	# for the main gui script to check whether an alert is triggered. returns appropriate boolean value
	def alert(self,):
		return self.alert_mode
	
	# retrieve message contents. will be tailored to specific emotion detected in last recorded event
	def get_event_alert(self,):
		if self.alert_mode and len(self.events) > 0:
			last_event = self.events[-1]  # last event recorded
			last_event_emo = last_event[2]
			bad = last_event[3]
			alert_msg = None

			if (last_event_emo == "Sadness") or (last_event_emo == "Anger"):
				alert_msg = "It appears you have been steadily expressing %s over the past 30 seconds. Maybe you'd benefit from taking a break?" % last_event_emo
			elif last_event_emo == "Overall Sentiment":
				alert_msg = "Your overall sentiment appears to have changed over the past 30 seconds. Is everything alright?"
			else:
				alert_msg = ""
			
			return alert_msg

	# Request that the camera captures a new image
	def request_new_img(self,):
		captured, frame = self.video_capture.read()
		if not captured:
			frame = cv2.imread("assets/no_camera.png")

		self.run_ml_on_img(frame)
		img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		img = Image.fromarray(img).resize((512, 288))
		img = ImageTk.PhotoImage(image=img)

		return img

	def get_event_img(self, i):
		if len(self.events) > i:
			return self.events[i][1] # stuff stored in self.events must be already ImageTk
		else:
			img = Image.fromarray(np.zeros((512, 288, 3)).astype(np.uint8)).resize((512, 288))
			return ImageTk.PhotoImage(image=img)

	def get_event_text(self, i):
		if len(self.events) > i:
			emotion = self.events[i][2]
			bad = {False: "improved", True: "worsened"}[self.events[i][3]]
			return "%s %s over the last 30 seconds" % (emotion, bad)
		else:
			return ""

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
		plt.close()
		fig, ax1 = plt.subplots(figsize=(7, 2.6))
		fig.patch.set_facecolor("#EBEBEB")

		if live:
			sent_data = np.convolve(np.array(self.emotion_mem)[-240:, 0], np.ones((3,)), mode="valid")/3
			ax1.plot(0 - np.array(range(len(sent_data)))[::-1], sent_data, color="tab:blue", linewidth=3)
		else:
			sent_data = np.convolve(np.array(self.emotion_mem)[:, 0], np.ones((3,)), mode="valid")/3
			ax1.plot(range(len(sent_data)), sent_data, color="tab:blue", linewidth=3)

		ax1.set_xlabel("Time (s)")
		# ax1.set_ylabel("Overall", color="tab:blue")
		ax1.tick_params(axis='y')
		ax1.set_yticks([-3, 0, 3], ["V. Poor", "Avg", "V. Good"], rotation="vertical", verticalalignment="center")
		ax1.set_ylim(-3.2, 3.2)

		# ax2 = ax1.twinx()
		# ax2.set_ylabel("Individual Emotions", color="black")
		# ax2.set_ylim(0, 3)
		#
		# for i in range(6):
		# 	label = ["Happiness", "Sadness", "Anger", "Fear", "Disgust", "Suprise"][i]
		# 	color = ["tab:green", "tab:olive", "tab:orange", "tab:purple", "tab:brown", "tab:pink"][i]
		# 	if live:
		# 		sent_data = np.convolve(np.array(self.emotion_mem)[-240:, i + 1], np.ones((10,)), mode="same")/10
		# 		ax2.plot(0 - np.array(range(len(sent_data)))[::-1], sent_data, label=label, color=color)
		# 	else:
		# 		sent_data = np.convolve(np.array(self.emotion_mem)[:, i + 1], np.ones((10,)), mode="same")/10
		# 		ax2.plot(range(len(sent_data)), sent_data, label=label, color=color)
		# ax2.legend(loc="upper left")

		if not live and len(self.events) > 0:
			for i, event in enumerate(self.events):
				if i == cur_event:
					ax1.vlines(event[0], -3, 3, color="tab:red", linewidth=5)
				else:
					ax1.vlines(event[0], -3, 3, color="black", linewidth=5)

		plt.tight_layout()

		buf = io.BytesIO()
		fig.savefig(buf, format='png')
		buf.seek(0)
		img = Image.open(buf).resize((700, 260))
		img = ImageTk.PhotoImage(image=img)
		buf.close()
		return img


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
	da = DataAgg()
