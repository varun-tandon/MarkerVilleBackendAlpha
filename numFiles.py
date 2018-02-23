import os, glob
i = 0
os.chdir("/Users/gmachiraju/Desktop/files")
for folder in glob.glob("*"):
    os.chdir("/Users/gmachiraju/Desktop/files/" + str(folder))
    for file in glob.glob("*"):
        i += 1
print i