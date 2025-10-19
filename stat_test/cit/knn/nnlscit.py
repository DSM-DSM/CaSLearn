from scipy import special, spatial
import pandas as pd
import random
import xgboost as xgb
import lightgbm as lgb
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def split_XYZ(data, dx, dy):
    X = data[:, 0:dx]
    Y = data[:, dx:dx + dy]
    Z = data[:, dx + dy:]
    return X, Y, Z


def split_train_test(data):
    total_size = data.shape[0]
    train_size = int(2 * total_size / 3)
    data_train = data[0:train_size, :]
    data_test = data[train_size:, :]
    return data_train, data_test


def normalize_data(data):
    data_norm = (data - np.mean(data, axis=0)) / (np.std(data, axis=0))
    return data_norm


def gen_bootstrap(data):
    np.random.seed()
    random.seed()
    num_samp = data.shape[0]
    I = np.random.permutation(num_samp)
    data_new = data[I, :]
    return data_new


eps = 1e-8


def mimic_knn(data_mimic, dx, dy, dz, Z_marginal):
    _, Y_train, Z_train = split_XYZ(data_mimic, dx, dy)
    nbrs = NearestNeighbors(n_neighbors=1).fit(Z_train)
    indx = nbrs.kneighbors(Z_marginal, return_distance=False).flatten()
    Y_marginal = Y_train[indx, :]
    return Y_marginal


def shuffle_y(data, dx):
    X = data[:, 0:dx]
    Y = data[:, dx:]
    Y = np.random.permutation(Y)
    return np.hstack((X, Y))


def log_mean_exp_numpy(fx_q, ax=0):
    eps = 1e-8
    max_ele = np.max(fx_q, axis=ax, keepdims=True)
    return (max_ele + np.log(eps + np.mean(np.exp(fx_q - max_ele), axis=ax, keepdims=True))).squeeze()


class Classifier_MI:

    def __init__(self, data_train_joint, data_eval_joint, data_train_marginal, data_eval_marginal,
                 dx, h_dim=64, actv=nn.ReLU, batch_size=32,
                 optimizer='adam', lr=0.001, max_ep=20, device=torch.device("cpu")):

        self.dim_x = dx
        self.data_dim = data_train_joint.shape[1]
        self.train_size = len(data_train_joint)
        self.eval_size = len(data_eval_joint)

        self.data_train_joint = data_train_joint
        self.data_train_marginal = data_train_marginal
        self.data_eval_joint = data_eval_joint
        self.data_eval_marginal = data_eval_marginal

        self.h_dim = h_dim
        self.actv = actv

        self.batch_size = batch_size
        self.optimizer_type = optimizer
        self.lr = lr
        self.max_iter = int(max_ep * self.train_size / batch_size)
        self.reg_coeff = 1e-3
        self.tol = 1e-4
        self.eps = 1e-8

        # Define model
        self.model = self._build_model()
        self.device = device
        self.model.to(self.device)

    def _build_model(self):
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, activation):
                super(MLP, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, 1)
                )
                # L2 regularization is handled via weight_decay in optimizer

            def forward(self, x):
                return self.net(x)

        return MLP(self.data_dim, self.h_dim, self.actv)

    def sample_pq_finite(self, batch_size):
        index = np.random.randint(low=0, high=self.train_size, size=batch_size)
        p_samples = self.data_train_joint[index]
        q_samples = self.data_train_marginal[index]
        return p_samples, q_samples

    def log_mean_exp_numpy(self, fx_q, ax=0):
        eps = 1e-8
        max_ele = np.max(fx_q, axis=ax, keepdims=True)
        return (max_ele + np.log(eps + np.mean(np.exp(fx_q - max_ele), axis=ax, keepdims=True))).squeeze()

    def train_classifier_MLP(self):

        criterion = nn.BCEWithLogitsLoss()
        if self.optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg_coeff)
        elif self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg_coeff)

        eval_inp_p = torch.tensor(self.data_eval_joint, dtype=torch.float32).to(self.device)
        eval_inp_q = torch.tensor(self.data_eval_marginal, dtype=torch.float32).to(self.device)
        B = len(eval_inp_p)

        for it in range(self.max_iter):
            # Sample and prepare batch
            batch_inp_p, batch_inp_q = self.sample_pq_finite(self.batch_size)
            batch_inp_p_tensor = torch.tensor(batch_inp_p, dtype=torch.float32).to(self.device)
            batch_inp_q_tensor = torch.tensor(batch_inp_q, dtype=torch.float32).to(self.device)
            batch_inp = torch.cat([batch_inp_p_tensor, batch_inp_q_tensor], dim=0)
            by = torch.cat([torch.ones((self.batch_size, 1)), torch.zeros((self.batch_size, 1))], dim=0)
            by = by.to(self.device)

            # Shuffle
            indices = torch.randperm(batch_inp.shape[0])
            batch_inp = batch_inp[indices]
            by = by[indices]

            # Forward pass
            logits = self.model(batch_inp)
            loss = criterion(logits, by)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final evaluation to get f values
        with torch.no_grad():
            finp_p = self.model(eval_inp_p).cpu().numpy()
            finp_q = self.model(eval_inp_q).cpu().numpy()

        rn_est_p = (1 / (1 + np.exp(-finp_p) + self.eps))
        finp_p = np.log(rn_est_p + self.eps) - np.log(1 - rn_est_p + self.eps)

        rn_est_q = (1 / (1 + np.exp(-finp_q) + self.eps))
        finp_q = np.log(rn_est_q + self.eps) - np.log(1 - rn_est_q + self.eps)

        div_est = np.mean(finp_p) - self.log_mean_exp_numpy(finp_q)

        return div_est


def xgb_classifier(joint_train_data, joint_test_data, marginal_train_data, marginal_test_data):
    data_train_feature = np.vstack((joint_train_data, marginal_train_data))
    data_train_label = np.vstack((np.ones((len(joint_train_data), 1)), np.zeros((len(marginal_train_data), 1))))
    data_index = np.random.permutation(2 * len(joint_train_data))
    data_train_feature = data_train_feature[data_index]
    data_train_label = data_train_label[data_index]

    data_test_feature = np.vstack((joint_test_data, marginal_test_data))
    data_test_label = np.vstack((np.ones((len(joint_test_data), 1)), np.zeros((len(marginal_test_data), 1))))
    data_test_index = np.random.permutation(2 * len(joint_test_data))
    data_test_feature = data_test_feature[data_test_index]
    data_test_label = data_test_label[data_test_index]

    model = xgb.XGBClassifier(
        # nthread=8,
        learning_rate=0.01,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.9,
        objective="binary:logistic",
        scale_pos_weight=1,
        seed=11,
        eval_metric="error",
    )

    gbm = model.fit(data_train_feature, data_train_label)

    y_pred_pos_prob = gbm.predict_proba(joint_test_data)
    y_pred_neg_prob = gbm.predict_proba(marginal_test_data)

    rn_est_p = (y_pred_pos_prob[:, 1] + eps) / (1 - y_pred_pos_prob[:, 1] - eps)
    finp_p = np.log(np.abs(rn_est_p))
    rn_est_q = (y_pred_neg_prob[:, 1] + eps) / (1 - y_pred_neg_prob[:, 1] - eps)
    finp_q = np.log(np.abs(rn_est_q))

    div_est = np.mean(finp_p) - log_mean_exp_numpy(finp_q)

    return div_est


def lgb_classifier(joint_train_data, joint_test_data, marginal_train_data, marginal_test_data):
    data_train_feature = np.vstack((joint_train_data, marginal_train_data))
    data_train_label = np.vstack((np.ones((len(joint_train_data), 1)), np.zeros((len(marginal_train_data), 1))))
    data_index = np.random.permutation(2 * len(joint_train_data))
    data_train_feature = data_train_feature[data_index]
    data_train_label = data_train_label[data_index]
    data_train_label = pd.DataFrame(data_train_label[data_index]).values.ravel()

    data_test_feature = np.vstack((joint_test_data, marginal_test_data))
    data_test_label = np.vstack((np.ones((len(joint_test_data), 1)), np.zeros((len(marginal_test_data), 1))))
    data_test_index = np.random.permutation(2 * len(joint_test_data))
    data_test_feature = data_test_feature[data_test_index]
    data_test_label = data_test_label[data_test_index]

    model = lgb.LGBMClassifier(learning_rate=0.1, metric='l1', n_estimators=20, num_leaves=38)
    gbm = model.fit(data_train_feature, data_train_label)

    y_pred_pos_prob = gbm.predict_proba(joint_test_data)
    y_pred_neg_prob = gbm.predict_proba(marginal_test_data)

    rn_est_p = (y_pred_pos_prob[:, 1] + eps) / (1 - y_pred_pos_prob[:, 1] - eps)
    finp_p = np.log(np.abs(rn_est_p))
    rn_est_q = (y_pred_neg_prob[:, 1] + eps) / (1 - y_pred_neg_prob[:, 1] - eps)
    finp_q = np.log(np.abs(rn_est_q))

    div_est = np.mean(finp_p) - np.log(np.abs(np.mean(rn_est_q)))

    return div_est


def rf_classifier(joint_train_data, joint_test_data, marginal_train_data, marginal_test_data):
    data_train_feature = np.vstack((joint_train_data, marginal_train_data))
    data_train_label = np.vstack((np.ones((len(joint_train_data), 1)), np.zeros((len(marginal_train_data), 1))))
    data_index = np.random.permutation(2 * len(joint_train_data))
    data_train_feature = data_train_feature[data_index]
    data_train_label = data_train_label[data_index]

    data_test_feature = np.vstack((joint_test_data, marginal_test_data))
    data_test_label = np.vstack((np.ones((len(joint_test_data), 1)), np.zeros((len(marginal_test_data), 1))))
    data_test_index = np.random.permutation(2 * len(joint_test_data))
    data_test_feature = data_test_feature[data_test_index]
    data_test_label = data_test_label[data_test_index]

    model = RandomForestClassifier(
        # nthread=8,
        n_estimators=100,
        max_depth=5,
    )

    gbm = model.fit(data_train_feature, data_train_label)

    y_pred_pos_prob = gbm.predict_proba(joint_test_data)
    y_pred_neg_prob = gbm.predict_proba(marginal_test_data)

    rn_est_p = (y_pred_pos_prob[:, 1] + eps) / (1 - y_pred_pos_prob[:, 1] - eps)
    finp_p = np.log(np.abs(rn_est_p))
    rn_est_q = (y_pred_neg_prob[:, 1] + eps) / (1 - y_pred_neg_prob[:, 1] - eps)
    finp_q = np.log(np.abs(rn_est_q))
    div_est = np.mean(finp_p) - log_mean_exp_numpy(finp_q)

    return div_est


def NNCMI(x, y, z, x_dim, y_dim, z_dim, device, classifier='xgb', normalize=False):
    data = np.hstack((x, y, z))

    if normalize:
        data = normalize_data(data)

    mimic_size = int(len(data) / 2)
    data_mimic = data[0:mimic_size, :]
    data_mine = data[mimic_size:, :]
    X, Y, Z = split_XYZ(data_mine, x_dim, y_dim)

    Y_marginal = mimic_knn(data_mimic, x_dim, y_dim, z_dim, Z)
    data_marginal = np.hstack((X, Y_marginal, Z))

    data_train_joint, data_eval_joint = split_train_test(data_mine)
    data_train_marginal, data_eval_marginal = split_train_test(data_marginal)

    # In our case, we recommend using xgb.
    if classifier == 'xgb':
        cmi_est_t = xgb_classifier(data_train_joint, data_eval_joint, data_train_marginal, data_eval_marginal)
    elif classifier == 'lgb':
        cmi_est_t = lgb_classifier(data_train_joint, data_eval_joint, data_train_marginal, data_eval_marginal)
    elif classifier == 'rf':
        cmi_est_t = rf_classifier(data_train_joint, data_eval_joint, data_train_marginal, data_eval_marginal)
    else:
        class_mlp_mi_xyz = Classifier_MI(data_train_joint, data_eval_joint, data_train_marginal, data_eval_marginal,
                                         x_dim, device)
        div_xyz_t = class_mlp_mi_xyz.train_classifier_MLP()
        cmi_est_t = div_xyz_t

    return cmi_est_t


def nnls_null_distribution(array, xyz, value, device, shuffle_neighbors=5, sig_samples=1000):
    dim, T = array.shape

    x_indices = np.where(xyz == 0)[0]
    y_indices = np.where(xyz == 1)[0]
    z_indices = np.where(xyz == 2)[0]

    seed = 42
    random_state = np.random.default_rng(seed)
    if len(z_indices) > 0 and shuffle_neighbors < T:

        z_array = array[z_indices, :].T.copy()
        tree_xyz = spatial.cKDTree(z_array)
        neighbors = tree_xyz.query(z_array,
                                   k=shuffle_neighbors,
                                   p=2,
                                   eps=0.)[1].astype(np.int32)

        null_dist = np.zeros(sig_samples)
        for sam in range(sig_samples):
            for i in range(len(neighbors)):
                random_state.shuffle(neighbors[i])
                # print('After randomly shuffling the k-nearest neighbor coordinates of zi, the neighbors are:')
            # print(neighbors)

            use_permutation = []
            for i in range(len(neighbors)):
                use_permutation.append(neighbors[i, 0])

            array_shuffled = np.copy(array)
            for i in x_indices:  # y_indices = [1]
                array_shuffled[i] = array[i, use_permutation]

            need_data = array_shuffled.T
            x0, y0, z0 = split_XYZ(need_data, dx=1, dy=1)
            x0_dim = x0.shape[1]
            y0_dim = y0.shape[1]
            z0_dim = z0.shape[1]
            null_dist[sam] = NNCMI(x0, y0, z0, x0_dim, y0_dim, z0_dim, device, classifier='xgb', normalize=False)

    pval = (1 + np.sum(null_dist >= value)) / (1 + sig_samples)

    return pval


def lpcmicit(x, y, z, classifier='xgb', normalize=False, shuffle_neighbors=5, sig_samples=200,
             device=torch.device("cpu")):
    x_dim = x.shape[1]
    y_dim = y.shape[1]
    z_dim = z.shape[1]
    real_cmi_value = NNCMI(x, y, z, x_dim, y_dim, z_dim, device=device, classifier=classifier, normalize=normalize)

    real_data = np.hstack((x, y, z))
    data = real_data.T
    xyz0 = np.array([0, 1] + [2 for i in range(z_dim)])

    p_value = nnls_null_distribution(array=data, xyz=xyz0, value=real_cmi_value, device=device,
                                     shuffle_neighbors=shuffle_neighbors, sig_samples=sig_samples)

    return p_value
