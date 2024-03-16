import os

try:
	os.mkdir("cmumosei")
except FileExistsError:
	print("Directory Exists")

# just_openface = {"OpenFace_2": mmdatasdk.cmu_mosi.highlevel["OpenFace_2"]}

# cmumosei_highlevel = mmdatasdk.mmdataset(just_openface, "cmumosei/")
# cmumosei_highlevel.add_computational_sequences(mmdatasdk.cmu_mosei.labels,'cmumosei/')
# cmumosei_highlevel.align('Opinion Segment Labels')

import gdown

gdown.download_folder(id="1A_hTmifi824gypelGobgl2M-5Rw9VWHv", output="cmumosei/")
