import json
import os
import sys

def update_performance_dictionary(performance_dic, mode, threshold, ps):
    if mode == "base":
        if ps < threshold:
            performance_dic["TN"] += 1
        else:
            performance_dic["FP"] += 1
    else:
        if ps < threshold:
            performance_dic["FN"] += 1
        else:
            performance_dic["TP"] += 1

    return performance_dic

def get_performance_metrics(performance_dic):
    #Accuracy
    accuracy = (performance_dic["TP"] + performance_dic["TN"]) / sum(performance_dic.values())

    #Precision
    precision = performance_dic["TP"] / (performance_dic["TP"] + performance_dic["FP"])

    #Recall
    recall = performance_dic["TP"] / (performance_dic["TP"] + performance_dic["FN"])

    #F1-Score
    f1 = (2 * performance_dic["TP"]) / ((2 * performance_dic["TP"]) + performance_dic["FP"] + performance_dic["FN"])

    return {"Accuracy":accuracy, "Precision":precision, "Recall":recall, "F1":f1}

def extract_core_name(filename):
    return filename[:-len("-base-test.txt")]


def model_threshold_eval(prop_file, base_file, prop_folder, base_folder):
    base_path = os.path.join(base_folder, base_file)
    prop_path = os.path.join(prop_folder, prop_file)
    base_content = open(base_path, "r", encoding="utf8")
    prop_content = open(prop_path, "r", encoding="utf8")
    base_scores = []
    prop_scores = []

    for line in base_content:
        if len(line.split()) > 0 and "PES articles:" not in line:
            line = "".join(line.split(":")).split()
            base_scores.append(float(line[-1]))

    for line in prop_content:
        if len(line.split()) > 0 and "PES articles:" not in line:
            line = "".join(line.split(":")).split()
            prop_scores.append(float(line[-1]))


    return base_scores[2], prop_scores[2]

def article_threshold_eval(prop_file, base_file, prop_folder, base_folder):
    base_path = os.path.join(base_folder, base_file)
    prop_path = os.path.join(prop_folder, prop_file)
    base_content = open(base_path, "r", encoding="utf8")
    prop_content = open(prop_path, "r", encoding="utf8")
    base_pes_scores = []
    prop_pes_scores = []

    for line in base_content:
        if "PES articles:" in line:
            line = line.split(":", 1)
            base_dic = json.loads(line[1].strip())
            base_pes_scores = base_dic.values()

    for line in prop_content:
        if "PES articles:" in line:
            line = line.split(":", 1)
            prop_dic = json.loads(line[1].strip())
            prop_pes_scores = prop_dic.values()

    return base_pes_scores, prop_pes_scores

if __name__ == "__main__":

    performance_dic = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    #Score folder paths
    prop_score_folder = os.path.join("eval", sys.argv[1])
    base_score_folder = os.path.join("eval", sys.argv[2])

    #Threshold to be evaluated
    threshold = float(sys.argv[3])

    #Determine evaluation mode (model or article)
    try:
        article_mode = bool(sys.argv[4])
    except:
        article_mode = False

    #List of score files
    prop_score_files = os.listdir(prop_score_folder)
    base_score_files = os.listdir(base_score_folder)

    #Bring lists into same order
    file_reorder = {extract_core_name(file): file for file in base_score_files}

    base_score_files = [file_reorder[extract_core_name(filename)] for filename in prop_score_files]

    #Update Accuracy Dictionary based on score files
    for psf, bsf in zip(prop_score_files, base_score_files):
        if article_mode:
            base_pes_scores, prop_pes_scores = article_threshold_eval(psf, bsf, prop_score_folder, base_score_folder)
            for base_pes, prop_pes in zip(base_pes_scores, prop_pes_scores):
                performance_dic = update_performance_dictionary(performance_dic, "base", threshold, base_pes)
                performance_dic = update_performance_dictionary(performance_dic, "prop", threshold, prop_pes)

        else:
            base_pes, prop_pes = model_threshold_eval(psf, bsf, prop_score_folder, base_score_folder)
            performance_dic = update_performance_dictionary(performance_dic, "base", threshold, base_pes)
            performance_dic = update_performance_dictionary(performance_dic, "prop", threshold, prop_pes)

    metric_dic = get_performance_metrics(performance_dic)

    #Write performance scores into file
    if article_mode:
        if not os.path.exists('eval/threshold_performance_article'):
            os.makedirs('eval/threshold_performance_article')

        with open("eval/threshold_performance_article/scores_{0}.txt".format(threshold), "w", encoding="utf8") as f:
            for key, value in zip(performance_dic.keys(), performance_dic.values()):
                f.write("{key}:       {value}\n".format(key=key, value=value))
            f.write("----------------------------------------------------------\n")
            for key, value in zip(metric_dic.keys(), metric_dic.values()):
                f.write("{key}:       {value:.4f}\n".format(key=key, value=value))
    else:
        if not os.path.exists('eval/threshold_performance_model'):
            os.makedirs('eval/threshold_performance_model')

        with open("eval/threshold_performance_model/scores_{0}.txt".format(threshold), "w", encoding="utf8") as f:
            for key, value in zip(performance_dic.keys(), performance_dic.values()):
                f.write("{key}:       {value}\n".format(key=key, value=value))
            f.write("----------------------------------------------------------\n")
            for key, value in zip(metric_dic.keys(), metric_dic.values()):
                f.write("{key}:       {value:.4f}\n".format(key=key, value=value))






