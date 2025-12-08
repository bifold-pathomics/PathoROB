import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer

from pathorob.robustness_index.robustness_index_utils import (compute_prediction_metrics,
    aggregate_stats, evaluate_embeddings, get_field_names_given_dataset, get_combi_meta_info, evaluate_knn_accuracy,
    get_k_values, save_balanced_accuracies, save_total_stats, plot_results_per_model, add_selected_metrics,
    calculate_per_class_prediction_stats
)


def select_optimal_k_value_pairs(dataset, model, patch_indices, embeddings, meta, results_folder, fig_folder, num_workers=8, DBG=False, compute_bootstrapped_robustness_index=False, opt_k=0, plot_graphs=True):
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
        print(f"select_optimal_k_value_pairs: project_combi {c+1}/{len(project_combis)} {project_combi}", flush=True)

        meta_combi = get_combi_meta_info(meta, patch_indices, embeddings, project_combi)
        if meta_combi is None:
            print(f'no patches found for project_combi {project_combi}; continuing')
            continue

        k_values = get_k_values(dataset, True, opt_k)
        if DBG and len(k_values) > 1:
            k_values = [k for k in k_values if k <= 51]  # limit k values for debugging

        X_scaled = np.vstack(meta_combi.embedding.values)
        bio_values = meta_combi[biological_class_field].values
        conf_values = meta_combi[confounding_class_field].values

        X_train = X_scaled
        X_test = X_scaled

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
            compute_prediction_metrics(dataset, meta_combi, stats, X_scaled, bio_values, conf_values)

            all_stats.append(stats)
        else:
            print(f"skipping {project_combi} train_classes {train_classes} test_classes {test_classes}")

    total_stats = aggregate_stats(all_stats, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index)
    #log to WandB here

    k_opt, bal_acc_at_k_opt, bio_class_prediction_result = save_balanced_accuracies(model, accuracies_bio, effective_k_values, results_folder)
    total_stats = save_total_stats(total_stats, meta, dataset, model, results_folder, k_opt, bal_acc_at_k_opt)
    calculate_per_class_prediction_stats(biological_class_field, confounding_class_field, bio_classes, model, meta, aucs_per_class_list, k_opt, results_folder)
    if plot_graphs:
        plot_results_per_model(total_stats, effective_k_values, model, fig_folder, dataset,
                               bio_class_prediction_result["bal_acc"], k_opt)
    return k_opt, bio_class_prediction_result, total_stats


def evaluate_model_pairs(dataset, data_manager, model, meta, results_folder, fig_folder, num_workers=8, k_opt_param = -1, DBG=False, compute_bootstrapped_robustness_index=False, plot_graphs=True):
    embeddings = data_manager.load_features(model, dataset, meta)
    print('loaded all embeddings')

    print(f"embeddings before normalize shape {embeddings.shape} max {np.max(embeddings)} min {np.min(embeddings)}")
    normalizer = Normalizer(norm='l2')
    embeddings = normalizer.fit_transform(embeddings)
    print(f"embeddings after normalize shape {embeddings.shape} max {np.max(embeddings)} min {np.min(embeddings)}")

    patch_indices = np.array(meta["patch_index"].values)

    print(f"len meta {len(meta)} patch_indices {len(patch_indices)} emb {len(embeddings)}")
    print("len index before ", len(meta.index), "unique", len(np.unique(meta.index)))

    k_opt, bio_class_prediction_results, robustness_metrics = select_optimal_k_value_pairs(
        dataset, model, patch_indices, embeddings, meta, results_folder, fig_folder,
        num_workers=num_workers, DBG=DBG, compute_bootstrapped_robustness_index=compute_bootstrapped_robustness_index,
        opt_k=k_opt_param, plot_graphs=plot_graphs
    )
    print(f"found k_opt {k_opt}")
    index_k_opt = list(robustness_metrics["k"]).index(k_opt)

    add_selected_metrics(model, robustness_metrics, bio_class_prediction_results, index_k_opt)

    return bio_class_prediction_results, robustness_metrics

