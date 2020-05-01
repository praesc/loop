import os
import sys

verbs = ['check', 'put']
numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
objects = ['sensor', 'nut', 'washer']
destinations = ['bag']

# Generate npz files
out_dir_npz = dir_path = sys.argv[1]
for idx in range(21):
    for verb in verbs:
        for number in numbers:
            for objectt in objects:
                for destination in destinations:
                    nexo = ' in the '
                    text = verb + ' ' + number + ' ' + objectt + nexo + destination
                    command = 'python3 generate_npz.py --text "' + text + '" --spkr ' + str(idx) + ' --checkpoint models/vctk/bestmodel.pth --out-dir ' + out_dir_npz
            os.system(command)

