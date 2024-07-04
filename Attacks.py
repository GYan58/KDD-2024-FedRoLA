from Settings import *
from Utils import *
from Defenses import *


def attk_Fang(re_para, k_paras, e_paras, g_num, a_paras):
    if g_num <= 1:
        return [re_para]

    direction = get_directions(k_paras, re_para)
    avg_k_paras = avg_params(k_paras)

    goal_ids = []
    find_para = None
    find_lambda = 0.01
    stop = False
    count = 0
    while not stop:
        r_para = cp.deepcopy(re_para)
        n_para = minus_params(r_para, find_lambda, direction)
        attack_paras = [n_para] * g_num + e_paras

        _, id = AggMKrum(attack_paras, g_num)

        if id in range(g_num):
            stop = True
            find_para = n_para
        else:
            find_lambda *= 0.5

        if find_lambda < 0.000001:
            stop = True
            find_para = n_para
        count += 1
    
    return [find_para] * g_num


def attk_MinMax(re_para, k_paras, g_num, a_paras):
    if g_num <= 1:
        return [re_para]

    direction = get_directions(k_paras, re_para)
    avg_k_paras = avg_params(k_paras)
    grads = [get_grad(k_paras[i], re_para) for i in range(len(k_paras))]
    avg_grads = avg_params(grads)

    max_dist = max(get_dist(grads[i], grads[j]) for i in range(len(k_paras)) for j in range(i, len(k_paras)))

    gamma = 0.001
    stop = False
    find_grad = None
    while not stop:
        n_grad = minus_params(avg_grads, gamma, direction)
        max_dist_curr = max(get_dist(n_grad, grads[i]) for i in range(len(k_paras)))

        if max_dist_curr < max_dist or gamma < 0.000001:
            stop = True
            find_grad = n_grad
        else:
            gamma *= 0.5

    find_para = minus_params(re_para, -1, find_grad)
    return [find_para] * g_num


def attk_MinSum(re_para, k_paras, g_num, a_paras):
    if g_num <= 1:
        return [re_para]

    direction = get_directions(k_paras, re_para)
    avg_k_paras = avg_params(k_paras)
    grads = [get_grad(k_paras[i], re_para) for i in range(len(k_paras))]
    avg_grads = avg_params(grads)

    max_dist = max(sum(get_dist(grads[i], grads[j]) for j in range(len(k_paras))) for i in range(len(k_paras)))

    gamma = 0.001
    stop = False
    find_grad = None
    while not stop:
        n_grad = minus_params(avg_grads, gamma, direction)
        total_dist = sum(get_dist(n_grad, grads[i]) for i in range(len(k_paras)))

        if total_dist < max_dist or gamma < 0.000001:
            stop = True
            find_grad = n_grad
        else:
            gamma *= 0.5

    find_para = minus_params(re_para, -1, find_grad)
    return [find_para] * g_num


def attk_Lie(re_para, k_paras, g_num, a_num, a_paras):
    if g_num <= 1:
        return [re_para]

    n = g_num
    m = int(n * 0.25)
    s = int(n / 2 + 1) - m
    z = st.norm.ppf((n - m - s) / (n - m))

    grads = [get_grad(k_paras[i], re_para) for i in range(len(k_paras))]
    avg_grads = avg_params(grads)
    direction = get_directions(k_paras, re_para)

    f_grad = cp.deepcopy(avg_grads)
    keys = k_paras[0].keys()
    for key in keys:
        if "bias" in key or "weight" in key:
            grad_values = np.array([grads[i][key].cpu().detach().numpy() for i in range(len(grads))])
            mu = np.mean(grad_values, axis=0)
            std = np.std(grad_values, axis=0)
            dir_values = direction[key].cpu().detach().numpy()

            result = mu + z * std * (dir_values < 0) - z * std * (dir_values > 0)
            f_grad[key] = torch.from_numpy(result).to(device)

    find_para = minus_params(re_para, -1, f_grad)
    return [find_para] * g_num


class MPHM:
    def __init__(self):
        self.last_b_grad = None

    def get_sigma(self, grads):
        vecs = [np.concatenate([grad[key].cpu().detach().numpy().reshape(-1) for key in grad if "weight" in key or "bias" in key]) for grad in grads]
        sigmas = np.std(vecs, axis=0)
        norm = np.linalg.norm(sigmas)
        return norm

    def attk_mphm(self, re_para, k_paras, g_num, a_paras):
        if g_num <= 1:
            return [re_para]

        direction = get_directions(k_paras, re_para)
        avg_k_paras = avg_params(k_paras)
        grads = [get_grad(k_paras[i], re_para) for i in range(len(k_paras))]
        avg_grads = avg_params(grads)
        delta_avg_grads = minus_params(avg_grads, -0.5, self.last_b_grad) if self.last_b_grad else cp.deepcopy(avg_grads)

        sigma_norm = self.get_sigma(grads)
        grad_norm = get_norm(delta_avg_grads)
        lambda_val = sigma_norm / grad_norm
        find_grad = minus_params(avg_grads, lambda_val, delta_avg_grads)
        self.last_b_grad = cp.deepcopy(find_grad)

        find_para = minus_params(re_para, -1, find_grad)
        return [find_para] * g_num


class AdapSimAttk:
    def __init__(self):
        self.mask = None
        self.last_bad_update = None
        self.last_good_update = None
        self.re_para = None
        self.at_ratio_bad = 0.99
        self.at_ratio_good = 0.1

    def adjust_ratio(self, re_para):
        if self.mask:
            recv_update = minus_params(re_para, 1, self.re_para)
            now_bad_update = {key: ((self.mask[key] > 0) + (self.mask[key] < 0)) * recv_update[key] for key in self.mask}
            now_good_update = {key: (self.mask[key] == 0) * recv_update[key] for key in self.mask}
            gsim_bad = get_cos_sim(self.last_bad_update, now_bad_update)
            gsim_good = get_cos_sim(self.last_good_update, now_good_update)

            if np.random.rand() <= 0.05:
                self.at_ratio_bad += 0.1 if np.random.rand() <= 0.5 else -0.1
            else:
                self.at_ratio_bad += 0.1 if gsim_bad > 0.0 else -0.1

            self.at_ratio_bad = max(0.01, min(0.99, self.at_ratio_bad))

        return self.at_ratio_bad, self.at_ratio_good

    def attk_sim(self, re_para, k_paras, g_num, a_paras):
        if g_num <= 1:
            return [re_para]

        at_ratio_bad, at_ratio_good = self.adjust_ratio(re_para)

        direction = get_top_k_directions(k_paras, re_para, at_ratio_bad)
        avg_k_paras = avg_params(k_paras)
        grads = [get_grad(k_paras[i], re_para) for i in range(len(k_paras))]
        a_grads = [get_grad(a_paras[i], re_para) for i in range(len(a_paras))]

        avg_grads = avg_params(grads)
        gamma = 0.000001
        gsim = 1

        while gsim > 0.1 and gamma <= 0.001:
            med_grad = minus_params(avg_grads, gamma, direction)
            gsim = get_cos_sim(med_grad, avg_grads)
            gamma *= 2

        min_dist = min(get_cos_sim(grads[i], grads[j]) for i in range(len(k_paras)) for j in range(i + 1, len(k_paras)))

        stop = False
        find_grad = None
        while not stop:
            n_grad = minus_params(avg_grads, gamma, direction)
            min_dist_curr = min(get_cos_sim(n_grad, grads[i]) for i in range(len(k_paras)))

            if min_dist_curr >= min_dist:
                stop = True
                find_grad = n_grad
            else:
                gamma *= 0.5

            if gamma < 0.000001:
                stop = True
                find_grad = n_grad

        bad_update = {key: find_grad[key] * ((direction[key] > 0) + (direction[key] < 0)) for key in direction}
        good_update = {key: find_grad[key] * (direction[key] == 0) for key in direction}
        self.last_bad_update = cp.deepcopy(bad_update)
        self.last_good_update = cp.deepcopy(good_update)
        self.re_para = cp.deepcopy(re_para)
        self.mask = cp.deepcopy(direction)

        find_para = minus_params(re_para, -1, find_grad)
        attack_paras = [get_dif_paras({key: (a_paras[i][key] - avg_grads[key] * at_ratio_good) * (direction[key] == 0) + find_para[key] * ((direction[key] > 0) + (direction[key] < 0)) for key in direction}, 0.01) for i in range(g_num)]

        return attack_paras
