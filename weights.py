import numpy as np


def get_class_weights(total_counts, class_positive_counts, multiply, use_class_balancing):
    """
    Calculate class_weight used in training

    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean 

    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    def balancing(class_weights, label_counts, multiply=10):
        """
        Normalize the class_weights so that each class has the same impact to backprop

        ex: label_counts: [1, 2, 3] -> factor: [1, 1/2, 1/3] * len(label_counts) / (1+1/2+1/3)
        """
        balanced = {}
        # compute factor
        reciprocal = np.reciprocal(label_counts.astype(float))
        factor = reciprocal * len(label_counts) * multiply / np.sum(reciprocal)

        # multiply by factor
        i = 0
        for c, w in class_weights.items():
            balanced[c] = {
                0: w[0] * factor[i],
                1: w[1] * factor[i],
            }
            i += 1
        return balanced

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = {}
    for i, class_name in enumerate(class_names):
        class_weights[class_name] = get_single_class_weight(label_counts[i], total_counts)

    if use_class_balancing:
        class_weights = balancing(class_weights, label_counts)
    return class_weights
