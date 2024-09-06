import os
import sys


def compare_scores(prop_file, base_file, prop_folder, base_folder):
    core_filename = extract_core_name(prop_file)
    print(prop_file)
    base_path = os.path.join(base_folder, base_file)
    prop_path = os.path.join(prop_folder, prop_file)
    base_content = open(base_path, "r", encoding="utf8")
    prop_content = open(prop_path, "r", encoding="utf8")
    base_scores = []
    prop_scores = []

    for line in base_content:
        if len(line.split()) > 0:
            line = "".join(line.split(":")).split()
            base_scores.append(float(line[-1]))

    for line in prop_content:
        if len(line.split()) > 0:
            line = "".join(line.split(":")).split()
            prop_scores.append(float(line[-1]))

    base_content.close()
    prop_content.close()
    if not os.path.exists(os.path.join("eval", "comparison")):
        os.makedirs(os.path.join("eval", "comparison"))

    with open(os.path.join("eval", "comparison" ,core_filename+".txt"), "w", encoding="utf8") as f:
        f.write("ΔAPD:    {0:.4f}\n".format(prop_scores[0] - base_scores[0]))
        f.write("Ratio:   {0:.4f}\n\n".format(prop_scores[0] / base_scores[0]))

        f.write("ΔPTD:    {0:.4f}\n".format(prop_scores[1] - base_scores[1]))
        f.write("Ratio:   {0:.4f}\n\n".format(prop_scores[1] / base_scores[1]))

        f.write("ΔPES:    {0:.4f}\n".format(prop_scores[2] - base_scores[2]))
        f.write("Ratio:   {0:.4f}".format(prop_scores[2] / base_scores[2]))

    print(prop_scores)
    print(base_scores)
    wait = input("Press Enter to continue.")
    return






def extract_core_name(filename):
    return filename[:-len("-base.txt")]



if __name__ == "__main__":



    prop_score_folder = os.path.join("eval", sys.argv[1])
    base_score_folder = os.path.join("eval", sys.argv[2])

    prop_score_files = os.listdir(prop_score_folder)
    base_score_files = os.listdir(base_score_folder)

    #Bring lists into same order
    file_reorder = {extract_core_name(file): file for file in base_score_files}
    base_score_files = [file_reorder[extract_core_name(filename)] for filename in prop_score_files]

    for psf, bsf in zip(prop_score_files, base_score_files):
        compare_scores(psf, bsf, prop_score_folder, base_score_folder)