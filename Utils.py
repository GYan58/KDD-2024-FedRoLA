from Models import *
from Settings import *

class RandomG:
    def __init__(self, Nclients=0):
        self.totalArms = OrderedDict()
        self.BadIDs = []
    
    def register_client(self, clientId):
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {}
            self.totalArms[clientId]['status'] = True
    
    def updateStatus(self,Id,Sta):
        self.totalArms[Id]['status'] = Sta
        
    def add_bad_clients(self, BIDs):
        self.BadIDs = BIDs
        for Id in self.BadIDs:
            self.totalArms[Id]['status'] = True
    
    def select_participants(self, num_of_clients):
        viable_clients = [x for x in self.totalArms.keys() if self.totalArms[x]['status']]
        return self.get_K(num_of_clients, viable_clients)
    
    def get_K(self, numOfSamples, feasible_clients):
        rd.shuffle(feasible_clients)
        pickedClients = feasible_clients[:numOfSamples]
        for ky in pickedClients:
            if ky in self.BadIDs:
                attackClients.append(ky)
                    
        return pickedClients, attackClients

def load_Model(Type, Name):
    Model = None
    model_mapping = {
        ("vgg", "cifar10"): (VGG_CIFAR10, "vgg_cifar10"),
        ("resnet", "cifar100"): (ResNet_CIFAR100, "res_cifar100"),
        ("alexnet", "fmnist"): (AlexNet_FMNIST, "alexnet_fmnist"),
        ("lstm", "shakespeare"): (Char_LSTM, "lstm_shake"),
        ("dnn", "harbox"): (DNN_HARBox, "dnn_harbox")
    }

    key = (Type, Name)
    if key in model_mapping:
        try:
            model_func, model_name = model_mapping[key]
            SPath = ModelRoot + model_name
            model_func = torch.load(SPath)
        except:
            model_func, model_name = model_mapping[key]
            SPath = ModelRoot + model_name
            torch.save(model_func, SPath)

    return model_func

def load_data_har(user_id):
    coll_class = []
    coll_label = []
    total_class = 0
    num_of_class = 5
    dimension_of_feature = 900
    class_set = ['Call', 'Hop', 'typing', 'Walk', 'Wave']

    for class_name in class_set:
        read_path = os.path.join(data_dir, f"{user_id}{Symbol}{class_name}_train.txt")
        if os.path.exists(read_path):
            temp_data = np.loadtxt(read_path).reshape(-1, 100, 10)[:, :, 1:10].reshape(-1, dimension_of_feature)
            count_img = temp_data.shape[0]
            coll_class.extend(temp_data)
            coll_label.extend([class_id] * count_img)
            total_class += 1

    return np.array(coll_class), np.array(coll_label), dimension_of_feature, total_class

def get_harbox():
    num_of_total_users = 120
    x_trains, x_tests, y_trains, y_tests = [], [], [], []

    for user_id in range(1, num_of_total_users + 1):
        x_coll, y_coll, _, _ = load_data_har(user_id)
        x_train, x_test, y_train, y_test = train_test_split(x_coll, y_coll, test_size=0.2, random_state=0)
        x_trains.extend(x_train)
        x_tests.extend(x_test)
        y_trains.extend(y_train)
        y_tests.extend(y_test)

    return (
        np.array([[[x]] for x in x_trains], dtype=float),
        np.array(y_trains, dtype=int),
        np.array([[[x]] for x in x_tests], dtype=float),
        np.array(y_tests, dtype=int)
    )

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def one_hot(index, size):
    vec = [0] * size
    vec[index] = 1
    return vec

def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

def word_to_indices(word):
    return [letter_to_index(c) for c in word]

def batch_data(data, batch_size):
    data_x = data['x']
    data_y = data['y']
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield batched_x, batched_y

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    for file in files:
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = sorted(data.keys())
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data

class ShakespeareDataset(Dataset):
    def __init__(self, data_dir, train=True):
        super(ShakespeareDataset, self).__init__()
        train_data_dir = os.path.join(data_dir, "train")
        test_data_dir = os.path.join(data_dir, "test")

        train_clients, train_groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

        self.train = train
        self.data = []
        self.labels = []
        self.client_data = {}

        if self.train:
            self._process_data(train_clients, train_data)
        else:
            self._process_data(train_clients, test_data)

    def _process_data(self, clients, data):
        for client_idx, client in enumerate(clients):
            self.client_data[client_idx] = set()
            client_x = data[client]['x']
            client_y = data[client]['y']
            
            for j in range(len(client_x)):
                self.client_data[client_idx].add(len(self.data))
                self.data.append(client_x[j])
                self.labels.append(client_y[j])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.labels[index]
        indices = word_to_indices(sentence)
        target_index = letter_to_index(target)
        return torch.LongTensor(indices), target_index

    def get_client_data(self):
        if self.train:
            return self.client_data
        else:
            raise ValueError("Test dataset does not contain client data.")
       
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_shakespeare_init(n_clients,dshuffle,batchsize):
    train_loader = ShakeSpeareDataset(train=True)
    test_loader = ShakeSpeareDataset(train=False)
    dict_users = train_loader.get_client_dic()
    dicts = []
    for ky in dict_users.keys():
        dicts += list(dict_users[ky])

    ELen = int(len(dicts) / n_clients)
    client_loaders = []
    for i in range(n_clients - 1):
        s_index = i * ELen
        e_index = (i + 1) * ELen
        new_dict = dicts[s_index:e_index]
        cloader = DataLoader(DatasetSplit(train_loader, new_dict), batch_size=batchsize, shuffle=dshuffle)
        client_loaders.append(cloader)

    cloader = DataLoader(DatasetSplit(train_loader, dicts[(n_clients - 1) * ELen:]), batch_size=batchsize, shuffle=dshuffle)
    client_loaders.append(cloader)
    
    train_loader = DataLoader(train_loader,batch_size=1000)
    test_loader = DataLoader(test_loader,batch_size=1000)

    return client_loaders, train_loader, test_loader

def get_shakespeare(n_clients,dshuffle,batchsize,partitions):
    train_loader = ShakeSpeare(train=True)
    test_loader = ShakeSpeare(train=False)
    
    client_loaders = []
    for i in range(n_clients):
        new_dict = partitions[i]
        cloader = DataLoader(DatasetSplit(train_loader, new_dict), batch_size=batchsize, shuffle=dshuffle)
        client_loaders.append(cloader)
        
    
    train_loader = DataLoader(train_loader,batch_size=1000)
    test_loader = DataLoader(test_loader,batch_size=1000)

    return client_loaders, train_loader, test_loader

def get_cifar10():
    data_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    TestX, TestY = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return TrainX, TrainY, TestX, TestY

def get_cifar100():
    data_train = torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
    data_test = torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    TestX, TestY = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return TrainX, TrainY, TestX, TestY

def get_fmnist():
    data_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)
    data_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)
    TrainX, TrainY = data_train.train_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    TestX, TestY = data_test.test_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)

    return TrainX, TrainY, TestX, TestY

class SplitData_Dirich:
    def __init__(self, dataset, labels, workers, balance=True, is_iid=True, alpha=0.0):
        self.dataset = dataset
        self.labels = labels
        self.workers = workers

        if alpha == 0 and not is_iid:
            raise ValueError("Non-IID splitting requires a positive alpha value.")

        if balance:
            partitions = [1 / workers] * workers
        else:
            partitions = self._generate_partitions(workers)

        if not is_iid and alpha > 0:
            self.partitions = self._get_dirichlet_partitions(labels, partitions, alpha)
        else:
            self.partitions = self._get_iid_partitions(labels, partitions)

    def _generate_partitions(self, workers):
        sum_partitions = workers * (workers + 1) / 2
        partitions = [(i + 1) / sum_partitions for i in range(workers)]
        partitions[-1] = 1 - sum(partitions[:-1])
        base_fraction = 0.1 / workers
        return [p * 0.9 + base_fraction for p in partitions]

    def _get_dirichlet_partitions(self, data, partitions, alpha):
        n_nets = len(partitions)
        n_classes = len(np.unique(self.labels))
        label_list = np.array(data)
        min_size = 0
        while min_size < n_classes:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(n_classes):
                idx_k = np.where(label_list == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                proportions = proportions * (len(idx_k) / n_nets)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min(len(idx_j) for idx_j in idx_batch)

        net_dataidx_map = {j: idx_batch[j] for j in range(n_nets)}

        return idx_batch

    def _get_iid_partitions(self, labels, partitions):
        data_len = len(labels)
        indices = list(range(data_len))
        rd.shuffle(indices)
        return [indices[int(sum(partitions[:i]) * data_len):int(sum(partitions[:i + 1]) * data_len)] for i in range(len(partitions))]

    def get_splits(self):
        clients_split = []
        for i in range(self.workers):
            indices = self.partitions[i]
            labels_split = self.labels[indices]
            data_split = self.dataset[indices]

            data_dict = defaultdict(list)
            for label, data in zip(labels_split, data_split):
                data_dict[label].append(data)

            Xs, Ys = [], []
            while sum(len(data_dict[label]) for label in data_dict) > 0:
                for label in data_dict:
                    if data_dict[label]:
                        Xs.append(data_dict[label].pop(0))
                        Ys.append(label)

            clients_split.append((np.array(Xs), np.array(Ys)))
            gc.collect()

        return clients_split

def get_data_transforms(name):
    transforms_dict = {
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    }
    return transforms_dict[name]

class CustomImageDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]

def get_loaders(name, n_clients=10, is_iid=True, alpha=0.5, batch_size=128):
    train_x, train_y, test_x, test_y = [], [], [], []
    
    if name == "fmnist":
        train_x, train_y, test_x, test_y = get_fmnist()
    elif name == "cifar10":
        train_x, train_y, test_x, test_y = get_cifar10()
    elif name == "cifar100":
        train_x, train_y, test_x, test_y = get_cifar100()
    elif name == "harbox":
        train_x, train_y, test_x, test_y = get_harbox()
    elif name == "shakespeare":
        client_loader, train_loader, test_loader = get_shakespeare_init(n_clients, False, batch_size)
        
        for inputs, targets in train_loader:
            train_x.extend(inputs.numpy() - 1e-8)
            train_y.extend(targets.numpy())
        for inputs, targets in test_loader:
            test_x.extend(inputs.numpy() - 1e-8)
            test_y.extend(targets.numpy())
        
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        
        splitter = SplitImageData(train_x, train_y, n_clients, True, is_iid, alpha)
        client_loaders, train_loader, test_loader = get_shakespeare(n_clients, False, batch_size, splitter.partitions)
        return client_loaders, train_loader, test_loader, max(test_y) + 1
    
    if name != "harbox":
        data_transforms = get_data_transforms(name)
    else:
        data_transforms = None

    splits = SplitImageData(train_x, train_y, n_clients, True, is_iid, alpha).get_splits()

    client_loaders = [
        torch.utils.data.DataLoader(CustomImageDataset(x, y, data_transforms), batch_size=batch_size, shuffle=True, drop_last=True)
        for x, y in splits
    ]

    train_loader = torch.utils.data.DataLoader(
        CustomImageDataset(train_x, train_y, data_transforms), batch_size=128, shuffle=False, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        CustomImageDataset(test_x, test_y, data_transforms), batch_size=128, shuffle=False, num_workers=2)

    return client_loaders, train_loader, test_loader, max(test_y) + 1

def extract_layer(params):
    return [param.flatten().unsqueeze(0) for ky, param in params.items() if "weight" in ky or "bias" in ky]

def cos_sim_vecs(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def two_norm_params(p1, p2):
    norm_sum = sum(np.linalg.norm(p1[ky].cpu().detach().numpy() - p2[ky].cpu().detach().numpy(), ord=2) ** 2 for ky in p1.keys() if "bias" in ky or "weight" in ky)
    return np.sqrt(norm_sum)

def one_norm_params(p1):
    norm_sum = sum(np.linalg.norm(p1[ky].cpu().detach().numpy(), ord=2) ** 2 for ky in p1.keys() if "bias" in ky or "weight" in ky)
    return np.sqrt(norm_sum)

def avg_params(params, lengths=None):
    if lengths is None:
        lengths = [1] * len(params)
    total_length = np.sum(lengths)
    result = cp.deepcopy(params[0])
    for ky in result.keys():
        result[ky] = sum(param[ky] * (length / total_length) for param, length in zip(params, lengths))
    return result

def minus_params(para1, multiplier, para2):
    result = cp.deepcopy(para1)
    for ky in result.keys():
        if "weight" in ky or "bias" in ky:
            result[ky] = para1[ky] - para2[ky] * multiplier
    return result

def get_grad(p1, p2):
    result = cp.deepcopy(p1)
    for ky in result.keys():
        if "weight" in ky or "bias" in ky:
            result[ky] = p1[ky] - p2[ky]
        else:
            result[ky].zero_()
    return result

def get_sim(w0, w1):
    norm0 = torch.tensor(0.0, device=w0[next(iter(w0))].device)
    norm1 = torch.tensor(0.0, device=w1[next(iter(w1))].device)
    dots = torch.tensor(0.0, device=w0[next(iter(w0))].device)

    for ky in w0.keys():
        if "weight" in ky or "bias" in ky:
            v0 = w0[ky]
            v1 = w1[ky]
            norm0 += torch.norm(v0) ** 2
            norm1 += torch.norm(v1) ** 2
            dots += torch.sum(v0 * v1)

    sim = dots / (torch.sqrt(norm0 * norm1))
    return sim.item()

def get_directions(params, reference_params, alpha=0.999, print_stats=False):
    avg_params_ = avg_params(params)
    grad = get_grad(avg_params_, reference_params)
    directions = cp.deepcopy(reference_params)
    thresholds = {ky: np.percentile(np.abs(grad[ky].cpu().detach().numpy()), (1 - alpha) * 100) for ky in grad.keys()}

    num_zeros, num_negatives, num_positives = 0, 0, 0
    for ky in grad.keys():
        grad_array = grad[ky].cpu().detach().numpy()
        direction_array = np.sign(np.where(np.abs(grad_array) > thresholds[ky], grad_array, 0))
        directions[ky] = torch.from_numpy(direction_array).to(device)

        num_zeros += np.sum(direction_array == 0)
        num_negatives += np.sum(direction_array == -1)
        num_positives += np.sum(direction_array == 1)

    if print_stats:
        total = num_zeros + num_negatives + num_positives
        print(f"* Stat of Directions: {num_negatives / total * 100:.2f}% -1, {num_zeros / total * 100:.2f}% 0, {num_positives / total * 100:.2f}% 1")

    return directions

def get_cos_sim(w0, w1):
    norm0 = 1e-6
    norm1 = 1e-6
    dots = 0
    for ky in w0.keys():
        if "weight" in ky or "bias" in ky:
            v0 = w0[ky].cpu()
            v1 = w1[ky].cpu()
            norm0 += torch.norm(v0) ** 2
            norm1 += torch.norm(v1) ** 2
            dots += torch.sum(v0 * v1)

    return dots / np.sqrt(norm0 * norm1)

def get_cos_sim_layer(w0, w1):
    sims = []
    for ky in w0.keys():
        if "weight" in ky:
            v0 = w0[ky].cpu()
            v1 = w1[ky].cpu()
            norm0 = torch.norm(v0)
            norm1 = torch.norm(v1)
            dots = torch.sum(v0 * v1)
            sims.append((dots / (norm0 * norm1)).item())
    return sims

def get_dist(w0, w1):
    dist = 0
    for ky in w0.keys():
        if "weight" in ky or "bias" in ky:
            dist += np.linalg.norm(w0[ky].cpu().detach().numpy() - w1[ky].cpu().detach().numpy()) ** 2
    return np.sqrt(dist)

def get_norm(weights, print_stats=False):
    norm_sum = sum(np.linalg.norm(weights[ky].cpu().detach().numpy()) ** 2 for ky in weights.keys() if "weight" in ky or "bias" in ky)
    if print_stats:
        for ky in weights.keys():
            if "weight" in ky or "bias" in ky:
                print(ky, np.linalg.norm(weights[ky].cpu().detach().numpy()) ** 2)
    return np.sqrt(norm_sum)

def get_dot_product(w0, w1):
    return sum(torch.sum(w0[ky].cpu() * w1[ky].cpu()) for ky in w0.keys() if "weight" in ky or "bias" in ky)

def get_different_params(params, variance=0.01):
    num_params = sum(param.cpu().reshape(-1).size for param in params.values() if "weight" in param or "bias" in param)
    new_params = cp.deepcopy(params)
    
    for ky in new_params.keys():
        factor = np.random.uniform(0.5, 1)
        delta = np.random.uniform(0, variance * factor) / np.sqrt(num_params)
        if "weight" in ky or "bias" in ky:
            new_params[ky] += delta
    
    return new_params

def get_top_k_directions(params, reference_params, k=0.5, print_stats=False):
    avg_params_ = avg_params(params)
    grad = get_grad(avg_params_, reference_params)
    directions = cp.deepcopy(reference_params)

    num_zeros, num_negatives, num_positives = 0, 0, 0
    for ky in grad.keys():
        grad_array = grad[ky].cpu().detach().numpy()
        threshold = np.percentile(np.abs(grad_array), (1 - k) * 100)
        direction_array = np.sign(np.where(np.abs(grad_array) > threshold, grad_array, 0))
        directions[ky] = torch.from_numpy(direction_array).to(device)

        num_zeros += np.sum(direction_array == 0)
        num_negatives += np.sum(direction_array == -1)
        num_positives += np.sum(direction_array == 1)

    if print_stats:
        total = num_zeros + num_negatives + num_positives
        print(f"* Stat of Directions: {num_negatives / total * 100:.2f}% -1, {num_zeros / total * 100:.2f}% 0, {num_positives / total * 100:.2f}% 1")

    return directions