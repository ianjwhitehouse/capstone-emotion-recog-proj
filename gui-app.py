from customtkinter import CTk, CTkFrame, CTkLabel, CTkButton
from tkinter import NONE, BOTH, X, TOP, BOTTOM, LEFT, RIGHT, RAISED
import numpy as np
from data_agg import DataAgg

# Start tkinter
root = CTk()
root.title("WorkMindfully")

# Start data agg
data_agg = DataAgg()

# Set window height
screenwidth = root.winfo_screenwidth()
screenheight = root.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (720, 640, (screenwidth - 720) / 2, (screenheight - 540) / 2)
root.geometry(alignstr)
root.resizable(width=False, height=False)

def generate_window(live_mode=True, draw_event=0):
	print(draw_event)
	print("window")
	for widget in root.winfo_children():
		widget.destroy()

	# Add camera frame
	camera_frame = CTkFrame(master=root, width=514, height=290) # , relief=RAISED, borderwidth=1)
	camera_frame.pack(side=TOP, padx=12, pady=6, fill=NONE, expand=False)
	camera_placement = CTkLabel(master=camera_frame, width=512, height=288, text="")
	camera_placement.pack(fill=NONE, expand=False)

	# Update label to set correct sizes
	camera_placement.update()

	# Method to draw new frames
	def put_image_into_frame():
		# Get a new image from the camera
		img = data_agg.request_new_img(camera_placement.winfo_width(), camera_placement.winfo_height())

		# Put image into frame
		# print("CAMERA", camera_placement.winfo_height(), camera_placement.winfo_width(), img.width(), img.height())
		camera_placement.photo_image=img
		camera_placement.configure(image=img)
		camera_placement.update()

		# Repeat
		camera_placement.after(1000//data_agg.cam_updates_per_second, put_image_into_frame)

	if live_mode:
		put_image_into_frame()
		
	else:
		# Get a screenshot saved with the specific event
		img = data_agg.get_event_img(draw_event, camera_placement.winfo_width(), camera_placement.winfo_height())

		# Put image into frame
		camera_placement.photo_image = img
		camera_placement.configure(image=img)
		camera_placement.update()

	# Add current emotion frame
	sent_frame = CTkFrame(master=root, height=24) # , relief=RAISED, borderwidth=1)
	sent_frame.pack(side=BOTTOM, padx=12, pady=6, fill=X, expand=False)

	sentiment_text = CTkLabel(master=sent_frame)
	sentiment_text.pack(side=LEFT, fill=X, padx=6, expand=False)

	# Function to draw sentiment
	def update_sent_text():
		sent = data_agg.get_emotion(0)

		if np.isnan(sent):
			sent = -1
		else:
			sent += 3
		# print("SENT", sent)

		# Put image into frame
		txt = "Overall sentiment: %s" % ["Very Poor", "Poor", "Below Average", "Average", "Above Average", "Good", "Very Good", "ND"][sent]
		sentiment_text.text = txt
		sentiment_text.configure(text=txt)

		# Repeat
		sentiment_text.after(1000, update_sent_text)

	if live_mode:
		update_sent_text()
	else:
		# sent = data_agg.get_event_emo(draw_event, 0)
		#
		# # Put image into frame
		# txt = "Overall: %s" % ["Very Poor", "Poor", "Below Average", "Average", "Above Average", "Good", "Very Good"][sent + 3]
		
		txt = data_agg.get_event_text(draw_event)
		
		sentiment_text.text = txt
		sentiment_text.configure(text=txt)

	# Add other emotions
	current_emo_frame = CTkFrame(master=root, height=24) # , relief=RAISED, borderwidth=1)
	current_emo_frame.pack(side=BOTTOM, padx=12, pady=6, fill=X, expand=False)

	emo_labels = ["Happiness", "Sadness", "Anger", "Fear", "Disgust", "Suprise"]

	# Function to draw emotions
	def update_emo_label(label_element, label_name, label_index):
		emo = data_agg.get_emotion(label_index)
		
		if np.isnan(emo):
			emo = -1
		# print("EMO", emo)

		# Put image into frame
		txt = "%s: %s" % (label_name, ["0 ", "1 ", "2 ", "3 ", "ND"][emo])
		label_element.text = txt
		label_element.configure(text=txt)

		# Repeat
		label_element.after(1000, lambda : update_emo_label(label_element, label_name, label_index))

	# Add all emotion labels
	for i in range(6):
		emo_label = CTkLabel(master=current_emo_frame, anchor="w")
		emo_label.pack(side=LEFT, fill=NONE, expand=False, padx=6)
		if i < 5:
			emo_sep = CTkLabel(master=current_emo_frame, text="|")
			emo_sep.pack(side=LEFT, fill=NONE, expand=False)
		if live_mode:
			update_emo_label(emo_label, emo_labels[i], i+1)
		else:
			emo = data_agg.get_event_emo(draw_event, i + 1)

			# Put image into frame
			txt = "%s: %s" % (
				emo_labels[i],
				["0", "1", "2", "3"][emo]
			)

			emo_label.text = txt
			emo_label.configure(text=txt)

	# Add button
	if live_mode:
		button = CTkButton(
			master=sent_frame, text="Stop Rec.", width=64, # relief=RAISED, borderwidth=1,
			command=lambda : generate_window(live_mode=False, draw_event=0)
		)
		button.pack(side=RIGHT, padx=6, fill=NONE, expand=False)
	else:
		button = CTkButton(
			master=sent_frame, text="Start Rec.", width=64, # relief=RAISED, borderwidth=1,
			command=lambda: generate_window(live_mode=True)
		)
		button.pack(side=RIGHT, padx=6, fill=NONE, expand=False)
			
		button = CTkButton(
			master=sent_frame, text="Load Rec.", width=64, # relief=RAISED, borderwidth=1,
			command=None
		)
		button.pack(side=RIGHT, padx=6, fill=NONE, expand=False)
		button = CTkButton(
			master=sent_frame, text="Save Rec.", width=64, # relief=RAISED, borderwidth=1,
			command=None
		)
		button.pack(side=RIGHT, padx=6, fill=NONE, expand=False)
		
		if len(data_agg.events) > 0:
			button = CTkButton(
				master=sent_frame, text="Next Event", width=64, # relief=RAISED, borderwidth=1,
				command=lambda: generate_window(live_mode=False, draw_event=min(draw_event + 1, len(data_agg.events) - 1))
			)
			button.pack(side=RIGHT, padx=6, fill=NONE, expand=False)
			
			button = CTkButton(
				master=sent_frame, text="Prev. Event", width=64, # relief=RAISED, borderwidth=1,
				command=lambda: generate_window(live_mode=False, draw_event=max(draw_event - 1, 0))
			)
			button.pack(side=RIGHT, padx=6, fill=NONE, expand=False)

	# Draw graph
	graph_frame = CTkFrame(master=root, width=702, height=262) #, relief=RAISED, borderwidth=1)
	graph_frame.pack(side=BOTTOM, padx=12, pady=6, fill=NONE, expand=False)
	graph_placement = CTkLabel(master=graph_frame, width=700, height=260, text="")
	graph_placement.pack(fill=NONE, expand=False)

	# Update label to set correct sizes
	graph_placement.update()

	# Method to draw new frames
	def put_graph_into_frame():
		img = data_agg.get_graph(graph_placement.winfo_width(), graph_placement.winfo_height())

		# Put image into frame
		graph_placement.photo_image = img
		graph_placement.configure(image=img)
		graph_placement.update()

		# Repeat
		graph_placement.after(1000, put_graph_into_frame)

	if live_mode:
		put_graph_into_frame()
		
	else:
		print("GRAPH", graph_placement.winfo_width(), graph_placement.winfo_height())
		img = data_agg.get_graph(graph_placement.winfo_width(), graph_placement.winfo_height(), live=False, cur_event=draw_event)

		# Put image into frame
		graph_placement.photo_image = img
		graph_placement.configure(image=img)
		graph_placement.update()
	
	print("done gening")


# Run app
generate_window()
root.mainloop()
del data_agg
