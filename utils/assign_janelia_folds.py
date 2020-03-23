import sys
import os
import random

files = []
for f in os.listdir("."):
    if f[-3:] == ".h5":
        files.append(f)

random.shuffle(files)

for i, f in enumerate(files):
    if i < 35:
        os.system("mv " + f + " train")
    elif i < 38:
        os.system("mv " + f + " val")
    elif i < 42:
        os.system("mv " + f + " test")


print("Done.")

