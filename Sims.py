from Utils import *
from Settings import *
from Defenses import *

class ClientSim:
    def __init__(self, train_loader, model, lr, weight_decay, epochs=1, n_classes=10):
        self.train_data = cp.deepcopy(train_loader)
        self.model = cp.deepcopy(model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_classes = n_classes
        self.dlen = len(self.train_data)
        self.round = 0

    def reload_traindata(self, loader):
        self.train_data = cp.deepcopy(loader)

    def get_params(self):
        return cp.deepcopy(self.model.state_dict())

    def update_params(self, params):
        self.model.load_state_dict(params)

    def update_lr(self, lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)

    def get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def local_train(self, dattk=None):
        self.model.train()
        for _ in range(self.epochs):
            for inputs, targets in self.train_data:
                if dattk == "Label":
                    targets = self.n_classes - targets - 1
                
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
        
        gc.collect()
        torch.cuda.empty_cache()
        self.scheduler.step()

    def evaluate(self, loader=None):
        self.model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()

        loader = loader or self.train_data

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == targets).sum().item()
                loss += loss_fn(outputs, targets).item()
                samples += targets.size(0)
                iters += 1

        return correct / samples, loss / iters


class ServerSim:
    def __init__(self, train_loader, test_loader, model, lr, weight_decay=0, epochs=2, dataset_name="cifar10"):
        self.train_data = cp.deepcopy(train_loader)
        self.test_data = cp.deepcopy(test_loader)
        self.model = cp.deepcopy(model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.recv_params = []
        self.recv_lens = []

        self.afa = AFA()
        self.fedrola_lasi = FedRoLA_LASI()
        self.fedrola_pcsi = FedRoLA_PCSI()

    def reload_testdata(self, loader):
        self.test_data = cp.deepcopy(loader)

    def get_params(self):
        return cp.deepcopy(self.model.state_dict())

    def get_lr(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']

    def update_params(self, params):
        self.model.load_state_dict(params)

    def avg_params(self, params, lens):
        res = cp.deepcopy(params[0])
        total_len = np.sum(lens)
        for key in res.keys():
            res[key] = sum(params[i][key] * (lens[i] / total_len) for i in range(len(params)))
        return res

    def sync_params(self, defense_method, attack_ids=[], update_ids=[]):
        frac = min(len(update_ids) // 2 - 1, len(attack_ids))

        if defense_method == "FedRoLA-LASI":
            last_params = self.get_params()
            new_params, bad_ids = self.fedrola_lasi.detect(last_params, self.recv_params, self.recv_lens, update_ids)
        elif defense_method == "FedRoLA-PCSI":
            last_params = self.get_params()
            new_params, bad_ids = self.fedrola_pcsi.detect(last_params, self.recv_params, self.recv_lens, update_ids)
        elif defense_method == "MultiKrum":
            num = max(1, len(self.recv_params) - frac - 2)
            new_params, bad_ids = MultiKrum(self.recv_params, frac, num, update_ids)
        elif defense_method == "TrimMean":
            new_params, _ = TrimMean(self.recv_params, frac)
        elif defense_method == "AFA":
            last_params = self.get_params()
            new_params, bad_ids = self.afa.agg_params(update_ids, last_params, self.recv_params, self.recv_lens)
        elif defense_method == "cosDefense":
            last_params = self.get_params()
            new_params, bad_ids = cosDefense(last_params, self.recv_params, self.recv_lens, update_ids)

        self.update_params(new_params)
        self.recv_params = []
        self.recv_lens = []
        self.optimizer.step()
        self.scheduler.step()

    def recv_info(self, params, length):
        self.recv_params.append(params)
        self.recv_lens.append(length)

    def evaluate(self, loader=None):
        self.model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0

        loader = loader or self.train_data

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == targets).sum().item()
                loss += self.loss_fn(outputs, targets).item()
                samples += targets.size(0)
                iters += 1

        return loss / iters, correct / samples

    def save_model(self, path):
        torch.save(self.model, path)
