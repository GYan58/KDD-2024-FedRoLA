from Settings import *
from Utils import *

def MultiKrum(params, frac, num=1, uids=None):
    n = len(params)
    m = n - frac
    if m <= 1:
        m = n
    
    distances = defaultdict(dict)
    keys = params[0].keys()

    for i in range(n):
        param1 = params[i]
        for j in range(i, n):
            param2 = params[j]
            if i == j:
                distances[i][j] = 0.0
                continue
            distance = 0.0
            if j > i:
                for key in keys:
                    if "weight" in key or "bias" in key:
                        distance += torch.norm(param1[key] - param2[key]).item() ** 2
                distance = np.sqrt(distance)
                distances[i][j] = distance
                distances[j][i] = distance

    if num == 1:
        find_id, find_val = -1, 1e20
        for i in range(n):
            dist = sorted(distances[i].values())
            sum_dist = sum(dist[:m])
            if find_val > sum_dist:
                find_val = sum_dist
                find_id = i
        return params[find_id], [find_id]

    elif num >= 2:
        p_dict = {i: sum(sorted(val.values())[:m]) for i, val in distances.items()}
        s_dict = sorted(p_dict.items(), key=lambda x: x[1])

        good_params, good_ids = [], []
        for i in range(num):
            key = s_dict[i][0]
            good_ids.append(uids[key] if uids else key)
            good_params.append(params[key])
        bad_ids = sorted(set(range(n)) - set(good_ids))
        return avg_params(good_params), bad_ids

def TrimMean(params, frac):
    n = len(params)
    k = min(frac, int(n / 2) - 1)
    f_param = {key: torch.zeros_like(val) for key, val in params[0].items()}

    for key in f_param.keys():
        if "bias" in key or "weight" in key:
            all_params = torch.stack([params[i][key] for i in range(n)])
            sorted_params, _ = torch.sort(all_params, dim=0)
            trimmed_params = sorted_params[k:n - k]
            f_param[key] = torch.mean(trimmed_params, dim=0)
        else:
            f_param[key] = sum(params[i][key] for i in range(n)) / n

    return f_param, [-1, -1]

class AFA:
    def __init__(self):
        self.alphas = {}
        self.betas = {}

    def add(self, id):
        self.alphas[id] = 0.5
        self.betas[id] = 0.5

    def agg_params(self, ids, r_param, params, lens):
        for key in ids:
            if key not in self.alphas:
                self.add(key)

        local_grads = {ids[i]: get_grad(params[i], r_param) for i in range(len(params))}
        lens = {ids[i]: lens[i] for i in range(len(params))}
        pks = {key: self.alphas[key] / (self.alphas[key] + self.betas[key]) for key in ids}

        good_ids = ids.copy()
        bad_ids, epi, step = [], 0.5, 1

        while True:
            grads = [local_grads[key] for key in good_ids]
            grad_lens = [lens[key] * pks[key] for key in good_ids]
            grad_r = avg_params(grads, grad_lens)

            sims = {key: get_sim(local_grads[key], grad_r) for key in good_ids}
            a_sims = list(sims.values())

            mu, std, med = np.mean(a_sims), np.std(a_sims), np.median(a_sims)
            remove_ids = []

            for key in good_ids:
                sim = sims[key]
                if (mu < med and sim < med - std * epi) or (mu >= med and sim > med + std * epi):
                    bad_ids.append(key)
                    remove_ids.append(key)

            if not remove_ids:
                break

            good_ids = list(set(good_ids) - set(remove_ids))
            epi += step

        if len(bad_ids) >= len(ids) / 2:
            good_ids = ids

        good_grads = [local_grads[key] for key in good_ids]
        good_lens = [lens[key] * pks[key] for key in good_ids]
        grad_res = avg_params(good_grads, good_lens)

        res = minus_params(r_param, -1, grad_res)
        avg_params_ = avg_params(params)

        for key in res.keys():
            if "bias" not in key and "weight" not in key:
                res[key] = avg_params_[key]

        for key in good_ids:
            self.alphas[key] += 1
        for key in bad_ids:
            self.betas[key] += 1

        return res, bad_ids

def cosDefense(last_param, params, lens, uids):
    grads = [minus_params(params[i], 1, last_param) for i in range(len(params))]
    
    avg_grad = last_param
    avg_grad_layer = extract_layer(avg_grad)[-1]
    
    e_layers = [extract_layer(grads[i])[-1] for i in range(len(params))]

    sims = [F.cosine_similarity(avg_grad_layer, e_layer, dim=1).item() for e_layer in e_layers]

    min_val, max_val = np.min(sims), np.max(sims)
    norm_sims = [(sim - min_val) / (max_val - min_val) for sim in sims]

    threshold = np.mean(norm_sims)
    bad_ids = [i for i in range(len(norm_sims)) if norm_sims[i] >= threshold]

    good_params = [params[i] for i in range(len(uids)) if i not in bad_ids]
    good_lens = [lens[i] for i in range(len(uids)) if i not in bad_ids]

    result = avg_params(good_params, good_lens)
    return result, bad_ids


class FedRoLA_LASI:
    def __init__(self):
        self.alpha = {}
        self.beta = {}
        self.round = 0
        self.alpha_client = {}
        self.beta_client = {}

    def get_disc(self, round_num):
        h_param = 0.1
        return 2 / (1 + np.exp(-h_param * round_num)) - 1

    def get_prob(self):
        total_counts = np.array(list(self.alpha.values())) + np.array(list(self.beta.values()))
        probs = np.array(list(self.alpha.values())) / total_counts
        return probs / np.sum(probs)

    def detect(self, last_param, params, lens, uids):
        grads = [minus_params(para, 1, last_param) for para in params]
        avg_grad = avg_params(grads)
        avg_grad_layer = extract_layer(avg_grad)

        extracted_layers = [extract_layer(grad) for grad in grads]
        num_layer = len(extracted_layers[0])

        if not self.alpha:
            self.alpha = {i: 1 for i in range(num_layer)}
            self.beta = {i: 0 for i in range(num_layer)}

        for uid in uids:
            if uid not in self.alpha_client:
                self.alpha_client[uid] = 1
                self.beta_client[uid] = 1

        current_prob = self.get_prob()
        chosen_layers = np.random.choice(range(num_layer), 3, replace=False, p=current_prob)

        finds = {}
        for layer_idx in chosen_layers:
            sims = [F.cosine_similarity(avg_grad_layer[layer_idx], layer[layer_idx], dim=1) for layer in extracted_layers]

            threshold = 0.9
            bad_ids = []
            stop = False
            while not stop:
                find = [i for i, sim in enumerate(sims) if sim >= threshold]
                proportion = len(find) / len(uids)

                if proportion < 0.5:
                    bad_ids = find
                    stop = True
                    self.alpha[layer_idx] += 1

                if threshold <= 0:
                    bad_ids = []
                    stop = True
                    self.beta[layer_idx] += 1

                threshold -= 0.1

            for uid in bad_ids:
                finds[uid] = finds.get(uid, 0) + 1

        general_finds = finds
        finds = []
        vote_num = 3
        while not finds and vote_num >= 0:
            finds = [uids[key] for key in general_finds if general_finds[key] >= vote_num]
            vote_num -= 1

        for uid in uids:
            if uid in finds:
                self.beta_client[uid] += 1
            else:
                self.alpha_client[uid] += 1

        bad_ids = []
        pure_params, pure_lens = [], []
        for i, gid in enumerate(uids):
            prob = self.alpha_client[gid] / (self.alpha_client[gid] + self.beta_client[gid])
            if gid in finds:
                prob *= self.get_disc(self.round)
                bad_ids.append(gid)

            pure_params.append(params[i])
            pure_lens.append(lens[i] * prob)

        result = avg_params(pure_params, pure_lens)
        self.round += 1

        return result, bad_ids

class FedRoLA_PCSI:
    def __init__(self):
        self.alpha = {}
        self.beta = {}
        self.round = 0
        self.alpha_client = {}
        self.beta_client = {}

    def get_disc(self, round_num):
        h_param = 0.1
        return 2 / (1 + np.exp(-h_param * round_num)) - 1

    def get_prob(self):
        total_counts = np.array(list(self.alpha.values())) + np.array(list(self.beta.values()))
        probs = np.array(list(self.alpha.values())) / total_counts
        return probs / np.sum(probs)

    def detect(self, last_param, params, lens, uids):
        grads = [minus_params(para, 1, last_param) for para in params]
        extracted_layers = [extract_layer(grad) for grad in grads]
        num_layer = len(extracted_layers[0])

        if not self.alpha:
            self.alpha = {i: 1 for i in range(num_layer)}
            self.beta = {i: 0 for i in range(num_layer)}

        for uid in uids:
            if uid not in self.alpha_client:
                self.alpha_client[uid] = 1
                self.beta_client[uid] = 1

        current_prob = self.get_prob()
        chosen_layers = np.random.choice(range(num_layer), 3, replace=False, p=current_prob)

        finds = {}
        for layer_idx in chosen_layers:
            sims = np.zeros((len(extracted_layers), len(extracted_layers)))

            for i in range(len(extracted_layers)):
                for j in range(i + 1, len(extracted_layers)):
                    gsim = F.cosine_similarity(extracted_layers[i][layer_idx], extracted_layers[j][layer_idx], dim=1).item()
                    sims[i][j] = sims[j][i] = gsim

            mean_sims = [np.mean(sorted(sims[i], reverse=True)[:2]) for i in range(len(extracted_layers))]

            threshold = 0.9
            bad_ids = []
            stop = False
            while not stop:
                find = [i for i, sim in enumerate(mean_sims) if sim >= threshold]
                proportion = len(find) / len(uids)

                if proportion < 0.5:
                    bad_ids = find
                    stop = True
                    self.alpha[layer_idx] += 1

                if threshold <= 0:
                    bad_ids = []
                    stop = True
                    self.beta[layer_idx] += 1

                threshold -= 0.1

            for uid in bad_ids:
                finds[uid] = finds.get(uid, 0) + 1

        general_finds = finds
        finds = []
        vote_num = 3
        while not finds and vote_num >= 0:
            finds = [uids[key] for key, count in general_finds.items() if count >= vote_num]
            vote_num -= 1

        for uid in uids:
            if uid in finds:
                self.beta_client[uid] += 1
            else:
                self.alpha_client[uid] += 1

        bad_ids = []
        pure_params, pure_lens = [], []
        for i, gid in enumerate(uids):
            prob = self.alpha_client[gid] / (self.alpha_client[gid] + self.beta_client[gid])
            if gid in finds:
                prob *= self.get_disc(self.round)
                bad_ids.append(gid)

            pure_params.append(params[i])
            pure_lens.append(lens[i] * prob)

        result = avg_params(pure_params, pure_lens)
        self.round += 1

        return result, bad_ids
