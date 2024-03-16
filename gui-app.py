from tkinter import Tk, Frame, Label, Button
from tkinter import NONE, BOTH, X, TOP, BOTTOM, LEFT, RIGHT, RAISED
from data_agg import DataAgg

# Start tkinter
root = Tk()
root.title("Emotion Manager")

# Start data agg
data_agg = DataAgg()

# Set window height
screenwidth = root.winfo_screenwidth()
screenheight = root.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (720, 640, (screenwidth - 720) / 2, (screenheight - 540) / 2)
root.geometry(alignstr)
root.resizable(width=False, height=False)

def generate_window(live_mode=True, draw_event=0):
	print("window")
	for widget in root.winfo_children():
		widget.destroy()

	# Add camera frame
	camera_frame = Frame(root, width=514, height=290, relief=RAISED, borderwidth=1)
	camera_frame.pack(side=TOP, padx=11, pady=11, fill=NONE, expand=False)
	camera_placement = Label(camera_frame, width=512, height=288)
	camera_placement.pack(fill=NONE, expand=False)

	# Method to draw new frames
	def put_image_into_frame():
		img = data_agg.request_new_img()

		# Put image into frame
		camera_placement.photo_image=img
		camera_placement.configure(image=img)

		# Repeat
		camera_placement.after(1000, put_image_into_frame)

	if live_mode:
		put_image_into_frame()
	else:
		img = data_agg.get_event_img(draw_event)

		# Put image into frame
		camera_placement.photo_image = img
		camera_placement.configure(image=img)

	# Add current emotion frame
	sent_frame = Frame(root, height=24, relief=RAISED, borderwidth=1)
	sent_frame.pack(side=BOTTOM, padx=11, pady=2, fill=X, expand=False)

	sentiment_text = Label(sent_frame)
	sentiment_text.pack(side=LEFT, fill=X, expand=False)

	# Function to draw sentiment
	def update_sent_text():
		sent = data_agg.get_emotion(0)

		# Put image into frame
		txt = "Overall: %s" % ["Very Poor", "Poor", "Below Average", "Average", "Above Average", "Good", "Very Good"][sent + 3]
		sentiment_text.text = txt
		sentiment_text.configure(text=txt)

		# Repeat
		sentiment_text.after(1, update_sent_text)

	if live_mode:
		update_sent_text()
	else:
		sent = data_agg.get_event_emo(draw_event, 0)

		# Put image into frame
		txt = "Overall: %s" % ["Very Poor", "Poor", "Below Average", "Average", "Above Average", "Good", "Very Good"][sent + 3]
		sentiment_text.text = txt
		sentiment_text.configure(text=txt)

	# Add other emotions
	current_emo_frame = Frame(root, height=24, relief=RAISED, borderwidth=1)
	current_emo_frame.pack(side=BOTTOM, padx=11, pady=2, fill=X, expand=False)

	# Function to draw emotions
	def update_emo_label(label_element, label_name, label_index):
		emo = data_agg.get_emotion(label_index)

		# Put image into frame
		txt = "%s: %s |" % (label_name, ["No Dect.  ", "Weak Dect.", "Dect.     ", "Str. Dect."][emo])
		if i == 6:
			txt = txt[:-2]
		label_element.text = txt
		label_element.configure(text=txt)

		# Repeat
		label_element.after(1, lambda : update_emo_label(label_element, label_name, label_index))

	# Add all emotion labels
	for i in range(6):
		emo_label = Label(current_emo_frame)
		emo_label.pack(side=LEFT, fill=X, expand=False)
		if live_mode:
			update_emo_label(emo_label, ["Happiness", "Sadness", "Anger", "Fear", "Disgust", "Suprise"][i], i+1)
		else:
			emo = data_agg.get_event_emo(draw_event, i + 1)

			# Put image into frame
			txt = "%s: %s |" % (
				["Happiness", "Sadness", "Anger", "Fear", "Disgust", "Suprise"][i],
				["Not Dect.   ", "Weak Dect.  ", "Detected    ", "Strong Dect."][emo]
			)
			if i == 6:
				txt = txt[:-2]

			emo_label.text = txt
			emo_label.configure(text=txt)

	# Add button
	if live_mode:
		button = Button(
			sent_frame, text="Stop Recording", relief=RAISED, borderwidth=1,
			command=lambda : generate_window(live_mode=False, draw_event=0)
		)
		button.pack(side=RIGHT, padx=11, fill=NONE, expand=False)
	else:
		button = Button(
			sent_frame, text="Start Recording", relief=RAISED, borderwidth=1,
			command=lambda: generate_window(live_mode=True)
		)
		button.pack(side=RIGHT, padx=11, fill=NONE, expand=False)

		button = Button(
			sent_frame, text="Prev. Event", relief=RAISED, borderwidth=1,
			command=lambda: generate_window(live_mode=False, draw_event=max(draw_event - 1, 0))
		)
		button.pack(side=RIGHT, padx=11, fill=NONE, expand=False)

		button = Button(
			sent_frame, text="Next Event", relief=RAISED, borderwidth=1,
			command=lambda: generate_window(live_mode=False, draw_event=max(draw_event + 1, len(data_agg.events)))
		)
		button.pack(side=RIGHT, padx=11, fill=NONE, expand=False)

	# Draw graph
	graph_frame = Frame(root, width=702, height=262, relief=RAISED, borderwidth=1)
	graph_frame.pack(side=BOTTOM, padx=11, pady=11, fill=NONE, expand=False)
	graph_placement = Label(graph_frame, width=700, height=260)
	graph_placement.pack(fill=NONE, expand=False)

	# Method to draw new frames
	def put_graph_into_frame():
		img = data_agg.get_graph()

		# Put image into frame
		graph_placement.photo_image = img
		graph_placement.configure(image=img)

		# Repeat
		graph_placement.after(1000, put_graph_into_frame)

	if live_mode:
		put_graph_into_frame()
	else:
		img = data_agg.get_graph(live=False, cur_event=draw_event)

		# Put image into frame
		graph_placement.photo_image = img
		graph_placement.configure(image=img)
	print("done gening")


# Run app
generate_window()
root.mainloop()
del data_agg
