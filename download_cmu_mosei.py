import os
from mmsdk import mmdatasdk

try:
	os.mkdir("cmumosei")
except FileExistsError:
	print("Directory Exists")

just_openface = {"OpenFace_2": "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/visual/CMU_MOSEI_VisualOpenFace2.csd"}

try:
    cmumosei_highlevel = mmdatasdk.mmdataset(just_openface, "cmumosei/")
except RuntimeError:
    cmumosei_highlevel = mmdatasdk.mmdataset({"OpenFace_2", "cmumosei/CMU_MOSEI_VisualOpenFace2.csd"})
    
cmumosei_highlevel.add_computational_sequences(mmdatasdk.cmu_mosei.labels,'cmumosei/')
cmumosei_highlevel.align('All Labels')
cmumosei_dataset["highlevel"].hard_unify()

# import gdown

# gdown.download_folder(id="1A_hTmifi824gypelGobgl2M-5Rw9VWHv", output="cmumosei/")
