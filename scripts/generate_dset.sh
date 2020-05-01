TMP='generated_npz'

# Generate range of npz
source be3/bin/activate
#python3 scripts/gen_npz.py ${TMP} 
deactivate

# Generate range of wav
source be/bin/activate
python scripts/gen_wav.py ${TMP}
deactivate

# Clean up
#rm -r ${TMP}

