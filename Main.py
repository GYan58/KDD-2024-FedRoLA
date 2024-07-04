from Settings import *
from Sims import *
from Utils import *
from Attacks import *

class FLProc:
    def __init__(self, configs):
        self.data_name = configs["dname"]
        self.model_name = configs["mname"]
        self.num_clients = configs["nclients"]
        self.num_participants = configs["pclients"]
        self.is_iid = configs["isIID"]
        self.alpha = configs["alpha"]
        self.max_iter = configs["iters"]
        self.log_step = configs["logstep"]
        self.lr = configs["learning_rate"]
        self.weight_decay = configs["weight_decay"]
        self.batch_size = configs["batch_size"]
        self.epoch = configs["epoch"]
        self.save_model = configs['save_model']
        self.attack = configs["attack"]
        self.defense_method = configs["defense"]
        self.num_control = int(configs["control_clients"])
        self.knowledge = configs["know"]
        self.diff_factors = configs["dif_factors"]
        
        self.server = None
        self.global_model = load_Model(self.model_name, self.data_name)

        self.clients = {}
        self.client_loaders = None
        self.train_loader = None
        self.test_loader = None
        self.num_classes = None
        
        self.update_ids = []
        self.attack_ids = []
        self.control_ids = list(range(self.num_control))
        self.train_round = 0

        self.selection = RandomG(self.num_clients)
        self.mphm_attk = MPHM()
        self.adap_sim_attack = AdapSimAttk()

    def get_train_data(self):
        self.client_loaders, self.train_loader, self.test_loader, self.num_classes = get_loaders(self.data_name, self.num_clients, self.is_iid, self.alpha, self.batch_size)

    def main(self):
        self.get_train_data()

        self.server = ServerSim(self.train_loader, self.test_loader, self.global_model, self.lr, self.weight_decay, self.epoch, self.data_name)
        for client_id in range(self.num_clients):
            self.clients[client_id] = ClientSim(self.client_loaders[client_id], self.global_model, self.lr, self.weight_decay, self.epoch, self.num_classes)
            self.selection.register_client(client_id)
            if client_id in self.control_ids:
                self.selection.update_status(client_id, False)
        
        self.selection.add_bad_clients(self.control_ids)
        
        for it in range(self.max_iter):
            self.train_round = it + 1

            update_ids, attack_ids = self.selection.select_participants(self.num_participants)
            global_params = self.server.get_params()
            global_lr = self.server.get_lr()
            trans_params, trans_lens = [], []

            for client_id in update_ids:
                self.clients[client_id].update_lr(global_lr)
                self.clients[client_id].update_params(global_params)
                if self.attack != "Label":
                    self.clients[client_id].local_train()
                else:
                    self.clients[client_id].local_train(dattk="Label" if client_id in attack_ids else None)

                trans_params.append(self.clients[client_id].get_params())
                trans_lens.append(self.clients[client_id].dlen)

            know_params, extra_params, attack_params = [], [], []
            if self.knowledge == "Part":
                for i, trans_len in enumerate(trans_lens):
                    if update_ids[i] in attack_ids:
                        know_params.append(trans_params[i])

            if self.knowledge == "Full":
                know_params = cp.deepcopy(trans_params)
                for i, trans_len in enumerate(trans_lens):
                    if update_ids[i] not in attack_ids:
                        extra_params.append(trans_params[i])

            for i, trans_len in enumerate(trans_lens):
                if update_ids[i] in attack_ids:
                    attack_params.append(trans_params[i])

            gen_num = len(attack_ids)
            if self.attack == "MinMax":
                bad_params = attk_MinMax(global_params, know_params, gen_num, attack_params)
                self.update_parameters(trans_params, bad_params, update_ids, attack_ids, global_params)

            if self.attack == "MinSum":
                bad_params = attk_MinSum(global_params, know_params, gen_num, attack_params)
                self.update_parameters(trans_params, bad_params, update_ids, attack_ids, global_params)
            
            if self.attack == "MPHM":
                bad_params = self.mphm_attk.attk_mphm(global_params, know_params, gen_num, attack_params)
                self.update_parameters(trans_params, bad_params, update_ids, attack_ids, global_params, mphm=True)

            if self.attack == "SimAttack":
                bad_params = self.adap_sim_attack.attk_sim(global_params, know_params, gen_num, attack_params)
                self.update_parameters(trans_params, bad_params, update_ids, attack_ids, global_params)

            if self.attack == "No":
                attack_ids = []

            for i, trans_len in enumerate(trans_lens):
                self.server.recv_info(trans_params[i], trans_len)

            self.server.sync_params(self.defense_method, attack_ids, update_ids)
            self.update_ids = update_ids
            self.attack_ids = attack_ids

    def update_parameters(self, trans_params, bad_params, update_ids, attack_ids, global_params, mphm=False):
        count = 0
        for i, trans_param in enumerate(trans_params):
            if update_ids[i] in attack_ids:
                bad_up = two_norm_params(bad_params[count], self.init_global_params)
                ba_params = get_different_params(bad_params[count], self.diff_factors)
                trans_params[i] = ba_params
                count += 1
            else:
                pure_up = two_norm_params(trans_params[i], self.init_global_params)
        
        self.benign_updates, self.bad_updates = [], []
        for i, trans_param in enumerate(trans_params):
            bb_up = minus_params(trans_param, 1, global_params)
            if update_ids[i] in attack_ids:
                self.bad_updates.append(bb_up)
            else:
                self.benign_updates.append(bb_up)


if __name__ == '__main__':
    configs = {
        "epoch": 3,
        "nclients": 128,
        "pclients": 32,
        "control_clients": 32,
        "learning_rate": 0.01,
        "save_model": False,
        "isIID": False,
        "logstep": 2,
        "dname": "fmnist",
        "mname": "alexnet",
        "learning_rate": 0.01,
        "iters": 200,
        "know": "Full",
        "alpha": 0.5,
        "attack": "MinMax",
        "defense": "FedRoLA-PCSI",
        "weight_decay": 1e-5,
        "batch_size": 16,
        "dif_factors": 0.01,
    }

    fl_proc = FLProc(configs)
    fl_proc.main()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
