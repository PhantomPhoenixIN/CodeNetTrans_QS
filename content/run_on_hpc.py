import subprocess

print("Start running on HPC")

# For Codenet Dataset data collection: which_python = "/home/swaminathanj/LIV/env/bin/python"
# which_python = "/home/swaminathanj/TranslatorRectifier/env/bin/python"

which_python = "/home/swaminathanj/TransRectify_new/env/bin/python"

main_path = "/home/swaminathanj/TransRectify_New/content/files/translator_module_prediction_progressive_learning.py"

subprocess.run([which_python,main_path])

print("Done running on HPC")