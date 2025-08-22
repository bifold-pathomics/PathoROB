import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer

from pathorob.robustness_index.robustness_index_utils import aggregate_stats, evaluate_embeddings, \
    get_field_names_given_dataset, get_combi_meta_info, evaluate_knn_accuracy, get_k_values, save_balanced_accuracies, \
    save_total_stats, plot_results_per_model, calculate_per_class_prediction_stats


def select_optimal_k_value_pairs(dataset, model, patch_names, embeddings, meta, results_folder, fig_folder, num_workers=8, DBG=False, compute_bootstrapped_robustness_index=False, opt_k=-1, plot_graphs=True):
    project_combis = np.unique(meta.subset.values)
    print(f"nr project_combis {len(project_combis)}")
    biological_class_field, confounding_class_field = get_field_names_given_dataset(dataset)
    print(f"select_optimal_k_value_pairs: len meta: {len(meta)}")

    bio_values = meta[biological_class_field].values
    bio_classes = np.unique(bio_values)

    accuracies_bio = []
    aucs_per_class_list = []
    if DBG:
        project_combis=project_combis[:2]
    all_stats = []
    for c, project_combi in enumerate(project_combis):
        print(f"select_optimal_k_value_pairs: project_combi {c}/{len(project_combis)} {project_combi}", flush=True)

        meta_combi = get_combi_meta_info(meta, patch_names, embeddings, project_combi)
        if meta_combi is None:
            print(f'no patches found for project_combi {project_combi}; continuing')
            continue

        max_samples_per_group = int(np.max(meta_combi["slide_id"].value_counts().values))

        k_values = get_k_values(dataset, opt_k, max_samples_per_group)
        if DBG and len(k_values) > 1:
            k_values = [k for k in k_values if k <= 51]  # limit k values for debugging

        X_unscaled = np.vstack(meta_combi.embedding.values)
        bio_values = meta_combi[biological_class_field].values

        X_train = X_unscaled
        X_test = X_unscaled

        y_train = bio_values
        y_test = bio_values

        nr_samples = len(X_train)
        k_values_sel = [k for k in k_values if k <= nr_samples]
        effective_k_values = []
        accuracies_k_bio = []

        train_classes = set(y_train)
        test_classes = set(y_test)

        knn_distances, knn_indices = None, None
        if train_classes == test_classes:
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            aucs_per_class = {}

            for k in k_values_sel[::-1]:
                #two-fold cross-validation per project_combi: train on half of WSIs, eval on other half, vice versa
                acc_score_bio, auc_per_class, knn_distances, knn_indices, effective_n_neighbors = evaluate_knn_accuracy(meta_combi, dataset, X_train, X_test, y_train, y_test, k, num_workers, knn_distances, knn_indices)
                effective_k_values.append(effective_n_neighbors)
                accuracies_k_bio.append(float(acc_score_bio))
                aucs_per_class[effective_n_neighbors] = auc_per_class

            effective_k_values = np.array(effective_k_values[::-1])  # reverse to match k_values_sel order
            accuracies_k_bio = accuracies_k_bio[::-1]  # reverse to match k_values_sel order
            accuracies_bio.append(accuracies_k_bio)
            aucs_per_class_list.append(aucs_per_class)

            stats = evaluate_embeddings(dataset, meta_combi, knn_indices)
            all_stats.append(stats)
        else:
            print(f"skipping {project_combi} train_classes {train_classes} test_classes {test_classes}")

    total_stats = aggregate_stats(all_stats, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index)
    #log to WandB here

    k_opt, bal_acc_at_k_opt, bio_class_prediction_result = save_balanced_accuracies(model, accuracies_bio, effective_k_values, results_folder)
    save_total_stats(total_stats, meta, dataset, model, results_folder, k_opt, bal_acc_at_k_opt)
    calculate_per_class_prediction_stats(biological_class_field, confounding_class_field, bio_classes, model, meta, aucs_per_class_list, k_opt, results_folder)
    if plot_graphs:
        plot_results_per_model(meta, total_stats, accuracies_bio, effective_k_values, model, results_folder, fig_folder, dataset,
                               bio_class_prediction_result["bal_acc"], k_opt)
    return k_opt, bio_class_prediction_result, total_stats

def evaluate_model_pairs(dataset, data_manager, model, meta, results_folder, fig_folder, num_workers=8, k_opt_param = -1, DBG=False, compute_bootstrapped_robustness_index=False, plot_graphs=True):
    embeddings = data_manager.load_features(dataset, meta)
    print('loaded all embeddings')

    print(f"embeddings before normalize shape {embeddings.shape} max {np.max(embeddings)} min {np.min(embeddings)}")
    normalizer = Normalizer(norm='l2')
    embeddings = normalizer.fit_transform(embeddings)
    print(f"embeddings after normalize shape {embeddings.shape} max {np.max(embeddings)} min {np.min(embeddings)}")

    patch_names = np.array(meta["patch_name"].values)

    print(f"len meta {len(meta)} patch_names {len(patch_names)} emb {len(embeddings)}")
    print("len index before ", len(meta.index), "unique", len(np.unique(meta.index)))

    k_opt, bio_class_prediction_results, robustness_metrics = select_optimal_k_value_pairs(
        dataset, model, patch_names, embeddings, meta, results_folder, fig_folder,
        num_workers=num_workers, DBG=DBG, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index,
        opt_k=k_opt_param, plot_graphs=plot_graphs
    )
    print(f"found k_opt {k_opt}")

    return bio_class_prediction_results, robustness_metrics


def calc_rob_index_pairs(data_manager, models, dataset, meta, results_folder, fig_folder, num_workers=8, k_opt_param=-1, DBG=False, compute_bootstrapped_robustness_index=False, plot_graphs=True):
    results = {}
    robustness_metrics = {}
    # TODO make this clean
    embeddings_folder = data_manager.features_dir
    for m,model in enumerate(models):
        fn = os.path.join(results_folder, f'frequencies-same-class-{model}.pkl')
        if os.path.exists(fn):
            print(f"model {model}: results already exist --> skipping. Found {fn}")
            continue
        # TODO make this clean
        data_manager.features_dir = embeddings_folder / model
        bio_class_prediction_results, robustness_metrics[model] = evaluate_model_pairs(dataset, data_manager, model, meta, results_folder, fig_folder, num_workers=num_workers, k_opt_param=k_opt_param, DBG=DBG, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index, plot_graphs=plot_graphs)
        results[model] = bio_class_prediction_results
    return results, robustness_metrics
