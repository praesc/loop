import os 
import sys

npz_path = sys.argv[1]
speakers = sorted(os.listdir(npz_path))
for speaker in speakers:
    speaker_idx = speaker[7:]
    file_path = os.path.join(npz_path, speaker)
    files = sorted(os.listdir(file_path))
    for fp in files:
        command = 'python generate_wav.py --npz ' + os.path.join(npz_path, speaker, fp) + ' --spkr ' + speaker_idx + ' --checkpoint models/vctk/bestmodel.pth --dataset-file synthetic-data.csv'
        os.system(command)
