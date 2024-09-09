import json
import os
import sys


# -------------------------HELPER FUNCTIONS----------------------------------- #

def get_technique_weights(folder_path):
    occ_lables = {}
    for filename in os.listdir(folder_path):
        if filename.startswith("article") and filename.endswith(".labels.tsv"):
            filepath = os.path.join(folder_path, filename)
            labels = pre_process_annotations(filepath)
            for label in labels:
                _, technique, _, _ = label
                if technique not in occ_lables:
                    occ_lables.update({technique: 0})
                occ_lables[technique] = occ_lables[technique] + 1

    total = sum(occ_lables.values())
    for key, value in occ_lables.items():
        weight = round(value / total, 7)
        occ_lables[key] = weight
    return occ_lables

def get_length_of_articles(folder_path):
    article_lengths = {}
    files = os.listdir(folder_path)

    for file_name in files:
        id = file_name[len("article"):-len(".txt")]

        with open(os.path.join(folder_path, file_name), "r", encoding="utf8") as f:
            article = f.read()
            article_lengths[id] = len(article)

    return article_lengths

def get_word_count_of_articles(folder_path):
    article_word_counts = {}
    for filename in os.listdir(folder_path):
        if filename.startswith("article") and filename.endswith(".txt"):
            article_id = filename[len("article"):-len(".txt")]
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf8") as file:
                content = file.read()
                words = content.split()
                article_word_counts[article_id] = len(words)

    return article_word_counts




def annotations_to_dic(data):
    technique_data = {}

    for entry in data:
        article_id, technique, start, end, word_count = entry
        if article_id not in technique_data:
            technique_data[article_id] = []
        technique_data[article_id].append({
            'technique': technique,
            'start': int(start),
            'end': int(end),
            'word_count': int(word_count)
        })

    return technique_data

def pre_process_annotations(filepath):
    split_annotations = []

    raw_annotations = open(filepath, "r", encoding="utf-8")

    for annotation in raw_annotations:
        split_annotation = annotation.strip("\n").split()
        split_annotations.append(split_annotation)

    raw_annotations.close()

    return split_annotations

# ----------------------------------------------------------------------------------------------- #

# ---------------------------- CALCULATE AVERAGE PROPAGANDA DENSITY ------------------------------------------------ #

def calc_article_prop_den(article_word_count, weights, article_annotations):
    pd_scores = []

    for a in article_annotations:
        technique = a['technique']
        weight = weights[technique]
        pd = (weight * a['word_count']) / int(article_word_count)
        pd_scores.append(pd)

    return sum(pd_scores)

def calc_average_prop_den(ids, annotations, word_counts, weights):
    article_pds = {}
    total_pd = 0

    for id in ids:
        article_pd = calc_article_prop_den(word_counts[id], weights, annotations[id])

        total_pd = total_pd + article_pd


    apd = total_pd/len(word_counts.keys())
    article_pds.update({'total': apd})

    return apd



# ------------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------- CALCULATE PROPAGANDA TECHNIQUE DIVERSITY ----------------------------------------- #

def calc_prop_tech_diversity(ids, num_articles,num_total_techniques, annotations):
    used_techniques = {}
    total = 0
    for id in ids:
        for annotation in annotations[id]:
            t = annotation["technique"]
            if id not in used_techniques:
                used_techniques[id] = []
            if t not in used_techniques[id]:
                used_techniques[id].append(t)
        total += len(used_techniques[id])/num_total_techniques


    ptd = round(total/num_articles, 4)
    return ptd
# --------------------------------------------------------------------------------------------------------------------- #

# ---------------------------CALCULATE PROPAGANDA EFFECTIVENESS SCORE-------------------------------------------------- #

def normalize_score_0_to_10(apd, ptd, max_weight):
    normalized_apd = (apd / max_weight) * 10
    normalized_ptd = ptd * 10

    return normalized_apd, normalized_ptd

def get_propaganda_effectiveness_score(APD, PTD):
    return (0.9 * APD) + (0.1* PTD)

# --------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------RUN CALCULATIONS--------------------------------------------------- #

def run_calculations(folder_path, weights, model, pred_file, mode):
    word_counts = get_word_count_of_articles(folder_path)
    pre_process = pre_process_annotations(pred_file)
    annotations = annotations_to_dic(pre_process)
    article_ids = list(word_counts.keys())
    pes_scores = {}

    # print(folder_path)
    # print("Weights: ",weights)
    # print("Word Counts: ", word_counts)
    # print("annotations: ", annotations)
    #wait = input("Press Enter to continue.")

    #Average Propaganda Density
    #apd, pd_scores = calc_average_prop_den(article_ids, annotations, word_counts, weights)

    #Propaganda Technique Diversity
    #ptd, ptd_scores = calc_prop_tech_diversity(article_ids, len(word_counts), len(weights.keys()), annotations)

    #Average Propaganda Density
    apd, pd_scores = calc_average_prop_den_test(article_ids, annotations, word_counts, weights)

    #Propaganda Technique Diversity
    ptd, ptd_scores = calc_prop_tech_diversity_test(article_ids, len(weights.keys()), annotations)

    #Normalize APD & PTD values to a range of 0-10
    norm_apd, norm_ptd = normalize_score_0_to_10(apd, ptd, max(weights.values()))




    #Total Propaganda Score
    total_count_propaganda_score = get_propaganda_effectiveness_score(norm_apd, norm_ptd)


    if not os.path.exists("eval/{folder}_scores".format(folder= mode)):
        os.makedirs("eval/{folder}_scores".format(folder = mode))
    if len(model.split("/")) > 1:
        model = model.split("/")[-1]

    for article_id in article_ids:
        norm_pd, norm_article_ptd = normalize_score_0_to_10(pd_scores[article_id], ptd_scores[article_id], max(weights.values()))
        pes = get_propaganda_effectiveness_score(norm_pd, norm_article_ptd)
        pes_scores.update({article_id: round(pes, 4)})


    with open("eval/{folder}_scores/{filename}.txt".format(folder = mode, filename= model), "w", encoding="utf8") as f:
        f.write("Normalized APD:            {0:.4f}\n".format(norm_apd))
        f.write("Normalized PTD:            {0:.4f}\n".format(norm_ptd))
        f.write("PES:                       {0:.4f}\n".format(total_count_propaganda_score))
        f.write("PES articles: {0}".format(json.dumps(pes_scores)))
    return total_count_propaganda_score, norm_apd, norm_ptd
# --------------------------------------------------------------------------------------------------------------------- #

def calc_average_prop_den_test(ids, annotations, word_counts, weights):
    article_pds = {}
    total_pd = 0

    for id in ids:
        article_pd = 0
        try:
            article_pd = calc_article_prop_den(word_counts[id], weights, annotations[id])
        except:
            article_pd = calc_article_prop_den(word_counts[id], weights, [])
        if id not in article_pds:
            article_pds.update({id: article_pd})
        total_pd = total_pd + article_pd


    apd = total_pd/len(word_counts.keys())


    article_pds.update({'total': apd})
    return apd, article_pds


def calc_prop_tech_diversity_test(ids, num_total_techniques, annotations):
    used_techniques = {}
    ptd_scores = {}
    total = 0
    for id in ids:
        if id not in ptd_scores:
            ptd_scores.update({id: 0})
        if id not in annotations.keys():
            continue
        for annotation in annotations[id]:
            t = annotation["technique"]
            if id not in used_techniques:
                used_techniques[id] = []
            if t not in used_techniques[id]:
                used_techniques[id].append(t)
        total += len(used_techniques[id])/num_total_techniques
        ptd_scores.update({id: len(used_techniques[id])/num_total_techniques})


    ptd = round(total/len(ids), 4)
    return ptd, ptd_scores

def update_accuracy_dictionary(accuracy_dic, mode, threshold, ps):
    if mode == "base":
        if ps < threshold:
            accuracy_dic["TN"] += 1
        else:
            accuracy_dic["FP"] += 1
    else:
        if ps < threshold:
            accuracy_dic["FN"] += 1
        else:
            accuracy_dic["TP"] += 1

    return accuracy_dic


def run_calculations_for_threshold(folder_path, weights, model, pred_file, mode, threshold):
    word_counts = get_word_count_of_articles(folder_path.format(model))
    pre_process = pre_process_annotations(pred_file)
    annotations = annotations_to_dic(pre_process)
    article_ids = list(word_counts.keys())

    accuracy_dic = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    ps_scores = {}

    #Average Propaganda Density
    apd, pd_scores = calc_average_prop_den_test(article_ids, annotations, word_counts, weights)

    #Propaganda Technique Diversity
    ptd, ptd_scores = calc_prop_tech_diversity_test(article_ids, len(weights.keys()), annotations)


    #Normalize APD & PTD values to a range of 0-10
    norm_apd, norm_ptd = normalize_score_0_to_10(apd, ptd, max(weights.values()))

    for article_id in article_ids:
        norm_pd, norm_article_ptd = normalize_score_0_to_10(pd_scores[article_id], ptd_scores[article_id], max(weights.values()))
        ps = get_propaganda_effectiveness_score(norm_pd, norm_article_ptd)
        accuracy_dic = update_accuracy_dictionary(accuracy_dic, mode, threshold, ps)
        ps_scores.update({article_id: round(ps, 4)})


    #Total Propaganda Score
    total_count_propaganda_score = get_propaganda_effectiveness_score(norm_apd, norm_ptd)


    return total_count_propaganda_score, accuracy_dic


def get_performance_metrics(accuracy_dic):
    #Accuracy
    accuracy = (accuracy_dic["TP"] + accuracy_dic["TN"]) / sum(accuracy_dic.values())

    #Precision
    precision = accuracy_dic["TP"] / (accuracy_dic["TP"] + accuracy_dic["FP"])

    #Recall
    recall = accuracy_dic["TP"] / (accuracy_dic["TP"] + accuracy_dic["FN"])

    #F1-Score
    f1 = (2 * accuracy_dic["TP"]) / ((2 * accuracy_dic["TP"]) + accuracy_dic["FP"] + accuracy_dic["FN"])

    return accuracy, precision, recall, f1



if __name__ == "__main__":
    train_folder = os.path.join("data/protechn_corpus_eval/train")

    article_folder = sys.argv[1]
    article_folder_path = os.path.join("data/protechn_corpus_eval/", article_folder)

    prediction_file = sys.argv[2]
    prediction_file_path = os.path.join("eval", prediction_file)

    mode = sys.argv[3]

    weights = get_technique_weights(train_folder)

    ps, apd, ptd = run_calculations(article_folder_path, weights, article_folder, prediction_file_path, mode)


