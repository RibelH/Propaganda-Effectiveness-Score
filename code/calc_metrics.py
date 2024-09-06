import os
import sys

thresholds = [0.38, 0.356, 0.35, 0.358, 0.36, 0.3634, 0.364, 0.3759]


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

    #Open file with annotations
    raw_annotations = open(filepath, "r", encoding="utf-8")

    # Strip and split annotations into a list
    for annotation in raw_annotations:
        split_annotation = annotation.strip("\n").split()
        split_annotations.append(split_annotation)

    raw_annotations.close()

    return split_annotations



def calc_article_prop_den(article_word_count, weights, article_annotations):

    count_scores = []
    for a in article_annotations:
        technique = a['technique']
        weight = weights[technique]
        apdt = (weight * a['word_count']) / int(article_word_count)
        count_scores.append(apdt)

    return sum(count_scores)

def calc_average_prop_den(ids, annotations, word_counts, weights):
    article_pds = {}
    total_pd = 0

    for id in ids:
        article_pd = calc_article_prop_den(word_counts[id], weights, annotations[id])
        total_pd = total_pd + article_pd


    apd = total_pd/len(word_counts.keys())

    article_pds.update({'total': apd})

    return apd

# def calc_prop_tech_diversity(ids, num_total_techniques, annotations):
#     used_techniques = []
#     for id in ids:
#         for annotation in annotations[id]:
#             t = annotation["technique"]
#             if t not in used_techniques:
#                 used_techniques.append(t)
#
#     num_used_techniques = len(used_techniques)
#     ptd = round(num_used_techniques / num_total_techniques, 4)
#     return ptd

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



def get_total_propaganda_score(APD, PTD):
    return (0.9 * APD) + (0.1* PTD)

#Normalizing Score to get
def normalize_score_0_to_10(apd, ptd, max_weight):
    normalized_apd = (apd / max_weight) * 10
    normalized_ptd = ptd * 10

    return normalized_apd, normalized_ptd


def calc_sentence_level_propaganda_density(word_counts, prop_words, ids):
    slpd = 0

    for id in ids:
        if id in prop_words.keys():
            slpd += prop_words[id] / word_counts[id]

    result = slpd / len(word_counts.keys())

    return result


def run_calculations(folder_path, weights, model, pred_file, mode):
    word_counts = get_word_count_of_articles(folder_path)
    pre_process = pre_process_annotations(pred_file)
    annotations = annotations_to_dic(pre_process)
    article_ids = annotations.keys()

    #Average Propaganda Density
    apd = calc_average_prop_den(article_ids, annotations, word_counts, weights)

    #Propaganda Technique Diversity
    ptd = calc_prop_tech_diversity(article_ids, len(word_counts), len(weights.keys()), annotations)

    #Normalize APD & PTD values to a range of 0-10
    norm_apd, norm_ptd = normalize_score_0_to_10(apd, ptd, max(weights.values()))

    #Total Propaganda Score
    total_count_propaganda_score = get_total_propaganda_score(norm_apd, norm_ptd)


    if not os.path.exists("eval/{folder}_scores".format(folder= mode)):
        os.makedirs("eval/{folder}_scores".format(folder = mode))
    if len(model.split("/")) > 1:
        model = model.split("/")[-1]



    with open("eval/{folder}_scores/{filename}.txt".format(folder = mode, filename= model), "w", encoding="utf8") as f:
        #f.write("+----------------------------PS-----------------------------+\n")
        #f.write("APD:                       {0:.4f}\n".format(apd))
        #f.write("PTD:                       {0}\n".format(ptd))
        f.write("Normalized APD:            {0:.4f}\n".format(norm_apd))
        f.write("Normalized PTD:            {0:.4f}\n".format(norm_ptd))
        f.write("PES:                       {0:.4f}\n\n".format(total_count_propaganda_score))

    return total_count_propaganda_score, norm_apd, norm_ptd

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
        ps = get_total_propaganda_score(norm_pd, norm_article_ptd)
        accuracy_dic = update_accuracy_dictionary(accuracy_dic, mode, threshold, ps)
        ps_scores.update({article_id: ps})


    #Total Propaganda Score
    total_count_propaganda_score = get_total_propaganda_score(norm_apd, norm_ptd)


    return total_count_propaganda_score, accuracy_dic

# def run_test_calculations(folder_path, weights, model, pred_file, mode):
#     word_counts = get_word_count_of_articles(folder_path.format(model))
#     pre_process = pre_process_annotations(pred_file)
#     annotations = annotations_to_dic(pre_process)
#     article_ids = annotations.keys()
#
#     #Average Propaganda Density
#     apd = calc_average_prop_den(article_ids, annotations, word_counts, weights)
#
#     #Propaganda Technique Diversity
#     ptd = calc_prop_tech_diversity(article_ids, len(word_counts), len(weights.keys()), annotations)
#
#     #Normalize APD & PTD values to a range of 0-10
#     norm_apd, norm_ptd = normalize_score_0_to_10(apd, ptd, max(weights.values()))
#
#     #Total Propaganda Score
#     total_count_propaganda_score = get_total_propaganda_score(norm_apd, norm_ptd)
#
#
#     if not os.path.exists("{folder}_scores".format(folder= mode)):
#         os.makedirs("{folder}_scores".format(folder = mode))
#     if len(model.split("/")) > 1:
#         model = model.split("/")[-1]
#
#
#
#     with open("{folder}_scores/{filename}.txt".format(folder = mode, filename= model), "w", encoding="utf8") as f:
#         f.write("+----------------------------PS-----------------------------+\n")
#         f.write("Normalized APD:            {0:.4f}\n".format(norm_apd))
#         f.write("Normalized PTD:            {0:.4f}\n".format(norm_ptd))
#         f.write("Total Propaganda Score:    {0:.4f}\n\n".format(total_count_propaganda_score))


def get_performance_metrics(final_accuracy_dic):
    #Accuracy
    accuracy = (final_accuracy_dic["TP"] + final_accuracy_dic["TN"]) / sum(final_accuracy_dic.values())

    #Precision
    precision = final_accuracy_dic["TP"] / (final_accuracy_dic["TP"] + final_accuracy_dic["FP"])

    #Recall
    recall = final_accuracy_dic["TP"] / (final_accuracy_dic["TP"] + final_accuracy_dic["FN"])

    #F1-Score
    f1 = (2 * final_accuracy_dic["TP"]) / ((2 * final_accuracy_dic["TP"]) + final_accuracy_dic["FP"] + final_accuracy_dic["FN"])

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


    # prop_folders = [
    #     "gpt-4o-prop",
    #     "gpt-4-turbo-prop",
    #     "gpt-4o-mini-prop",
    #     "gpt-3.5-turbo-prop",
    #     "gpt-3.5-turbo-instruct-prop",
    #     "meta-llama/Llama-2-7b-chat-hf-prop",
    #     "meta-llama/Meta-Llama-3-8B-Instruct-prop",
    #     "meta-llama/prop/Meta-Llama-3.1-8B-Instruct"
    # ]
    #
    # base_folders = [
    #     "gpt-4o-base",
    #     "gpt-4-turbo-base",
    #     "gpt-4o-mini-base",
    #     "gpt-3.5-turbo-base",
    #     "gpt-3.5-turbo-instruct-base",
    #     "meta-llama/Llama-2-7b-chat-hf-base",
    #     "meta-llama/Meta-Llama-3-8B-Instruct-base",
    #     "meta-llama/base/Meta-Llama-3.1-8B-Instruct"
    # ]
    #
    #
    # prediction_files_prop = [
    #     "official_prediction20240901-160120-gpt-4o-prop.txt",
    #     "official_prediction20240828-194132-gpt-4-turbo-prop.txt",
    #     "official_prediction20240829-013506-gpt-4o-mini-prop.txt",
    #     "official_prediction20240831-132929-gpt-3.5-turbo-prop.txt",
    #     "official_prediction20240829-000157-gpt-3.5-turbo-instruct-prop.txt",
    #     "official_prediction20240829-174434-Llama-2-prop.txt",
    #     "official_prediction20240829-174236-Llama-3-prop.txt",
    #     "official_prediction20240828-084849-Llama-3.1-prop.txt"
    #
    # ]
    #
    # prediction_files_base = [
    #     "official_prediction20240901-160034-gpt-4o-base.txt",
    #     "official_prediction20240828-194225-gpt-4-turbo-base.txt",
    #     "official_prediction20240829-013426-gpt-4o-mini-base.txt",
    #     "official_prediction20240831-132954-gpt-3.5-turbo-base.txt",
    #     "official_prediction20240829-000305-gpt-3.5-turbo-instruct-base.txt",
    #     "official_prediction20240829-174349-Llama-2-base.txt",
    #     "official_prediction20240829-174151-Llama-3-base.txt",
    #     "official_prediction20240828-084724-Llama-3.1-base.txt"
    # ]

    # prop_ps_scores = {}
    # base_ps_scores = {}
    # prop_apd_scores = {}
    # base_apd_scores = {}
    # prop_ptd_scores = {}
    # base_ptd_scores = {}
    # accuracy_dic = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    #
    # if not os.path.exists("threshold_performances_for_models"):
    #     os.makedirs("threshold_performances_for_models")

    # for model, pf in zip(prop_folders, prediction_files_prop):
    #     prop_ps, prop_apd, prop_ptd = run_calculations(article_folder_relative_path, weights, model, pf,"prop")
    #     if len(model.split("/")) > 1:
    #         model = model.split("/")[-1]
    #     prop_ps_scores.update({model: prop_ps})
    #     prop_apd_scores.update({model: prop_apd})
    #     prop_ptd_scores.update({model: prop_ptd})
    #
    #
    # for model, pf in zip(base_folders, prediction_files_base):
    #     base_ps, base_apd, base_ptd = run_calculations(article_folder_relative_path, weights, model, pf, "base")
    #     if len(model.split("/")) > 1:
    #         model = model.split("/")[-1]
    #     base_ps_scores.update({model: base_ps})
    #     base_apd_scores.update({model: base_apd})
    #     base_ptd_scores.update({model: base_ptd})


    # for model, pf in zip(base_folders, prediction_files_base):
    #     base_ps, base_apd, base_ptd = run_calculations(article_folder_relative_path, weights, model, pf, "base")
    #     if len(model.split("/")) > 1:
    #         model = model.split("/")[-1]
    #     base_ps_scores.update({model: base_ps})
    #     base_apd_scores.update({model: base_apd})
    #     base_ptd_scores.update({model: base_ptd})




    # for bmodel, pmodel in zip(base_folders, prop_folders):
    #     if len(pmodel.split("/")) > 1:
    #         bmodel = bmodel.split("/")[-1]
    #         pmodel = pmodel.split("/")[-1]
    #
    #     if not os.path.exists("comparison_ps"):
    #         os.makedirs("comparison_ps")
    #
    #
    #
    #     with open("./comparison_ps/{0}.txt".format("-".join(bmodel.split("-")[:-1])), "w", encoding="utf8") as f:
    #         f.write("ΔP:        {0:.4f}\n".format(prop_ps_scores[pmodel] - base_ps_scores[bmodel]))
    #         f.write("Ratio:     {0:.2f}\n".format(prop_ps_scores[pmodel] / base_ps_scores[bmodel]))
    #
    #     if not os.path.exists("comparison_apd"):
    #         os.makedirs("comparison_apd")
    #
    #     with open("./comparison_apd/{0}.txt".format(bmodel), "w", encoding="utf8") as f:
    #         f.write("ΔP:        {0:.4f}\n".format(prop_apd_scores[pmodel] - base_apd_scores[bmodel]))
    #         f.write("Ratio:     {0:.2f}\n".format(prop_apd_scores[pmodel] / base_apd_scores[bmodel]))
    #
    #     if not os.path.exists("comparison_ptd"):
    #         os.makedirs("comparison_ptd")
    #
    #     with open("./comparison_ptd/{0}.txt".format(bmodel), "w", encoding="utf8") as f:
    #         f.write("ΔP:        {0:.4f}\n".format(prop_ptd_scores[pmodel] - base_ptd_scores[bmodel]))
    #         f.write("Ratio:     {0:.2f}\n".format(prop_ptd_scores[pmodel] / base_ptd_scores[bmodel]))


    #Evaluating the thresholds based on the list of 20 news articles in test_articles.txt
    # file_names = [
    #     "official_prediction20240902-145350-gpt-3.5-turbo-instruct-base-test.txt",
    #     "official_prediction20240902-145440-gpt-3.5-turbo-instruct-prop-test.txt",
    #     "official_prediction20240902-145600-gpt-3.5-turbo-base-test.txt",
    #     "official_prediction20240902-145620-gpt-3.5-turbo-prop-test.txt",
    #     "official_prediction20240902-145745-gpt-4-turbo-base-test.txt",
    #     "official_prediction20240902-145807-gpt-4-turbo-prop-test.txt",
    #     "official_prediction20240902-145945-gpt-4o-mini-base-test.txt",
    #     "official_prediction20240902-150028-gpt-4o-mini-prop-test.txt",
    #     "official_prediction20240902-150129-gpt-4o-base-test.txt",
    #     "official_prediction20240902-150152-gpt-4o-prop-test.txt",
    #     "official_prediction20240903-134154-Llama-2-base-test.txt",
    #     "official_prediction20240903-134222-Llama-2-prop-test.txt",
    #     "official_prediction20240903-134328-Llama-3-base-test.txt",
    #     "official_prediction20240903-134351-Llama-3-prop-test.txt",
    #     "official_prediction20240903-134450-Llama-3.1-base-test.txt",
    #     "official_prediction20240903-134503-Llama-3.1-prop-test.txt"
    # ]
    #
    # test_article_folders = [
    #     "gpt-4o-{0}-test",
    #     "gpt-4-turbo-{0}-test",
    #     "gpt-4o-mini-{0}-test",
    #     "gpt-3.5-turbo-{0}-test",
    #     "gpt-3.5-turbo-instruct-{0}-test",
    #     "meta-llama/Llama-2-7b-chat-hf-{0}-test",
    #     "meta-llama/Meta-Llama-3-8B-Instruct-{0}-test",
    #     "meta-llama/Meta-Llama-3.1-8B-Instruct-{0}-test"
    # ]
    #
    # test_prop_articles = [folder.format("prop") for folder in test_article_folders]
    # test_base_articles = [folder.format("base") for folder in test_article_folders]
    #
    #
    #
    #
    #
    # # Files containing 'base'
    # test_base_files = [f for f in file_names if "base" in f]
    #
    # # Files containing 'prop'
    # test_prop_files = [f for f in file_names if "prop" in f]
    #
    # for model, pf in zip(test_prop_articles, test_prop_files):
    #     run_test_calculations(article_folder_relative_path, weights, model, pf,"prop_test")
    #
    # for model, pf in zip(test_base_articles, test_base_files):
    #     run_test_calculations(article_folder_relative_path, weights, model, pf, "base_test")



    # thresholds = [0.355,0.356,0.457,0.358, 0.36, 0.3634, 0.3759, 0.38]
    # base_scores = []
    # prop_scores = []
    # final_accuracy_dic = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    #
    # for threshold in thresholds:
    #     base_scores = []
    #     prop_scores = []
    #     for model, pf in zip(test_prop_articles, test_prop_files):
    #         ps, acc_dic = run_calculations_for_threshold(article_folder_relative_path, weights, model, pf, "prop", threshold)
    #         prop_scores.append(ps)
    #         final_accuracy_dic = {key: acc_dic.get(key, 0) + final_accuracy_dic.get(key, 0) for key in set(acc_dic) | set(final_accuracy_dic)}
    #
    #     for model, pf in zip(test_base_articles, test_base_files):
    #         ps, acc_dic = run_calculations_for_threshold(article_folder_relative_path, weights, model, pf, "base", threshold)
    #         final_accuracy_dic = {key: acc_dic.get(key, 0) + final_accuracy_dic.get(key, 0) for key in set(acc_dic) | set(final_accuracy_dic)}
    #         base_scores.append(ps)


    #     acc, prec, rec, f1 = get_performance_metrics(final_accuracy_dic)
    #
    #     if not os.path.exists("threshold_performances"):
    #         os.makedirs("threshold_performances")
    #
    #     with open("./threshold_performances/threshold_performance-{0}.txt".format(threshold), "w", encoding="utf8") as f:
    #
    #         f.write("Threshold: {0}\n".format(threshold))
    #         f.write("Accuracy:  {0:.4f}\n".format(acc))
    #         f.write("Recall:    {0:.4f}\n".format(rec))
    #         f.write("Precision: {0:.4f}\n".format(prec))
    #         f.write("F1-Score:  {0:.4f}\n".format(f1))
    #     print(final_accuracy_dic)
    #     final_accuracy_dic.update(dict.fromkeys(final_accuracy_dic, 0))
    #
    #
    # for threshold in thresholds:
    #     for ps in list(base_scores):
    #         final_accuracy_dic = update_accuracy_dictionary(final_accuracy_dic, "base", threshold, ps)
    #
    #     for ps in list(prop_scores):
    #         final_accuracy_dic = update_accuracy_dictionary(final_accuracy_dic, "prop", threshold, ps)
    #     acc, prec, rec, f1 = get_performance_metrics(final_accuracy_dic)
    #
    #     with open("./threshold_performances_for_models/threshold_performance-{0}.txt".format(threshold), "w", encoding="utf8") as f:
    #
    #         f.write("Threshold: {0}\n".format(threshold))
    #         f.write("Accuracy:  {0:.4f}\n".format(acc))
    #         f.write("Recall:    {0:.4f}\n".format(rec))
    #         f.write("Precision: {0:.4f}\n".format(prec))
    #         f.write("F1-Score:  {0:.4f}\n".format(f1))
    #
    #     final_accuracy_dic.update(dict.fromkeys(final_accuracy_dic, 0))


