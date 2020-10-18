import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper_fns.re_evaluate import re_evaluate_from_file


directory = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp_output"
)

seed = sys.argv[1]
func = sys.argv[2]
device = sys.argv[3]

# specify the parameters for the files to read
key_list = [
    "tts_apx_q=1",
    "tts_rhoKG_q=1",
    "one_shot_q=1",
    "apx_cvar_q=1",
    "random",
    "tts_apx_cvar_q=1",
]

low_fant_keys = ["tts_apx_q=1", "tts_rhoKG_q=1", "tts_apx_cvar_q=1"]

for output_key in key_list:
    if func == "bw":
        function_name = "braninwilliams"
        suffix = "_var_%s_" % output_key
        if output_key in low_fant_keys:
            suffix2 = "_low_fant_4_weights.pt"
        else:
            suffix2 = "_weights.pt"
    elif func == "bw_cvar":
        function_name = "braninwilliams"
        suffix = "_cvar_%s_" % output_key
        if output_key in low_fant_keys:
            suffix2 = "_low_fant_4_weights.pt"
        else:
            suffix2 = "_weights.pt"
    elif func == "m6":
        function_name = "marzat"
        suffix = "_cvar_%s_" % output_key
        if output_key in low_fant_keys:
            suffix2 = "_a=0.75_cont_low_fant_4.pt"
        else:
            suffix2 = "_a=0.75_cont.pt"
    elif func == "covid":
        function_name = "covid"
        suffix = "_cvar_%s_" % output_key
        if output_key in low_fant_keys:
            suffix2 = "_a=0.9_low_fant_4_weights.pt"
        else:
            suffix2 = "_a=0.9_weights.pt"
    elif func == "port":
        function_name = "portfolio_surrogate"
        suffix = "_var_%s_" % output_key
        if output_key in low_fant_keys:
            suffix2 = "_a=0.8_cont_low_fant_4.pt"
        else:
            suffix2 = "_a=0.8_cont.pt"
    else:
        raise ValueError("Unknown function!")

    filename = os.path.join(directory, function_name + suffix + str(seed) + suffix2)
    print("Starting file %s" % filename)
    re_evaluate_from_file(filename, function_name, device, True)
