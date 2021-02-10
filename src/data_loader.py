import collections
import os
import numpy as np
import torch


class DataLoader(object):
    def __init__(self, args):
        self.PAD_ID = 0
        self.OFFSET = 1

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hop = args.hop

        self.num_u2t = args.num_u2t
        self.num_i2t = args.num_i2t

        self.rating_np, self.n_user, self.n_item = self._load_rating()
        self.train_data, self.eval_data, self.test_data, self.user_items_dict, self.item_user_rating_dict = self._split_data()
        self.kg, self.n_entity, self.n_relation = self._load_kg()

        self.user_triples_dict = self._load_user_triples_dict()
        self.item_users_dict = self._load_item_users_dict()
        self.item_triples_dict = self._load_item_triples_dict()

        self.train_batches, self.eval_batches, self.test_batches = self._load_batches()

    def _load_user_triples_dict(self):
        user_triples_dict = collections.defaultdict(tuple)
        for user, seed_items in self.user_items_dict.items():
            seed_items = set(seed_items)
            heads, rels, tails = self._get_neighbors(seed_items, int(self.num_u2t / 2))
            user_triples_dict[user] = self._get_rel_loc(heads, rels, tails)

        return user_triples_dict

    def _load_item_users_dict(self):
        item_users_dict = collections.defaultdict(list)
        item_rating_dict = collections.defaultdict(list)
        for user, items in self.user_items_dict.items():
            for item in items:
                item_users_dict[item].append(user)
                item_rating_dict[item].append(self.item_user_rating_dict['i{}=u{}'.format(item, user)])

        sorted_item_users_dict = collections.defaultdict(list)
        for item, users in item_users_dict.items():
            ratings = item_rating_dict[item]
            unsorted = [ratings] + [range(0, len(users))]
            sort = [list(t) for t in zip(*sorted(zip(*unsorted)))]
            sorted_users = list(np.array(users)[sort[1]])
            sorted_item_users_dict[item] = sorted_users

        sparsity_items = set(range(1, self.n_item + 1)) - set(sorted_item_users_dict.keys())
        for item in sparsity_items:
            sorted_item_users_dict[item] = [0]

        return sorted_item_users_dict

    def _load_item_triples_dict(self):
        item_triples_dict = collections.defaultdict(tuple)
        for item in range(1, self.n_item + 1):
            seed_items = {item}
            heads, rels, tails = self._get_neighbors(seed_items, int(self.num_i2t / 2))
            item_triples_dict[item] = self._get_rel_loc(heads, rels, tails)

        return item_triples_dict

    def _get_neighbors(self, seed_items, sample_num):
        h_list, r_list, t_list = [], [], []
        for _ in range(self.hop):
            for item in seed_items:
                h_l = [item] * len(self.kg[item])
                r_l = [temp[0] for temp in self.kg[item]]
                t_l = [temp[1] for temp in self.kg[item]]

                h_list.extend(h_l)
                r_list.extend(r_l)
                t_list.extend(t_l)

            seed_items = set(t_list) - set(h_list)

        if len(h_list) > 0:
            replace = len(h_list) < sample_num
            indices = np.random.choice(len(h_list), size=sample_num, replace=replace)
            h_list = [h_list[i] for i in indices]
            r_list = [r_list[i] for i in indices]
            t_list = [t_list[i] for i in indices]

        return h_list, r_list, t_list

    def _get_rel_loc(self, heads, rels, tails):
        all_nodes = list(set(heads) | set(tails))
        node_size = len(all_nodes)

        rel_dict = collections.defaultdict(list)
        for h, r, t in zip(heads, rels, tails):
            h_idx, t_idx = all_nodes.index(h), all_nodes.index(t)
            rel_dict[r].append((h_idx, t_idx, h, t))

        rel_size = len(rel_dict.keys())

        rel_adj_dict = collections.defaultdict(tuple)
        for rel, l in rel_dict.items():
            h_array = np.zeros([node_size, node_size], dtype=int)
            r_array = np.zeros([node_size, node_size], dtype=int)
            t_array = np.zeros([node_size, node_size], dtype=int)
            for h_idx, t_idx, h, t in l:
                h_array[h_idx, t_idx] = h
                r_array[h_idx, t_idx] = rel
                t_array[h_idx, t_idx] = t

            rel_adj_dict[rel] = (h_array, r_array, t_array)

        return np.array(all_nodes), node_size, rel_adj_dict, rel_size

    def _load_batches(self):
        np.random.shuffle(self.train_data)
        np.random.shuffle(self.eval_data)
        np.random.shuffle(self.test_data)

        train_batches = []
        eval_batches = []
        test_batches = []

        total = int(self.train_data.shape[0] / self.batch_size) + 1
        start = 0
        while start + self.batch_size < self.train_data.shape[0]:
            packaged_batch = self.get_one_batch(self.train_data, start, start + self.batch_size)
            train_batches.append(packaged_batch)
            start += self.batch_size

        start = 0
        while start + self.batch_size < self.eval_data.shape[0]:
            packaged_batch = self.get_one_batch(self.eval_data, start, start + self.batch_size)
            eval_batches.append(packaged_batch)
            start += self.batch_size

        start = 0
        while start + self.batch_size < self.test_data.shape[0]:
            packaged_batch = self.get_one_batch(self.test_data, start, start + self.batch_size)
            test_batches.append(packaged_batch)
            start += self.batch_size

        return train_batches, eval_batches, test_batches

    def get_one_batch(self, data, start, end):
        user_batch = list(data[start: end, 0])
        item_batch = list(data[start: end, 1])
        label_batch = list(data[start: end, 2])

        batch_size = len(user_batch)

        u_i_batch, u_n_batch = [], []
        i_u_batch, i_n_batch = [], []

        for user in user_batch:
            u_i = self.user_items_dict[user]
            u_n, u_n_size, u_r_dict, u_r_size = self.user_triples_dict[user]

            u_i_batch.append(u_i)
            u_n_batch.append((u_n, u_n_size, u_r_dict, u_r_size))

        for item in item_batch:
            i_u = self.item_users_dict[item]
            i_n, i_n_size, i_r_dict, i_r_size = self.item_triples_dict[item]

            i_u_batch.append(i_u)
            i_n_batch.append((i_n, i_n_size, i_r_dict, i_r_size))

        u_max_len, u_i_len_batch, u_ori_idx_batch, user_batch, u_i_batch, u_n_batch = self._sorted_pack(batch_size, user_batch, u_i_batch, u_n_batch)
        i_max_len, i_u_len_batch, i_ori_idx_batch, item_batch, i_u_batch, i_n_batch = self._sorted_pack(batch_size, item_batch, i_u_batch, i_n_batch)

        u_ori_idx_batch = [t[1] for t in sorted(zip(*([u_ori_idx_batch] + [range(batch_size)])))]
        i_ori_idx_batch = [t[1] for t in sorted(zip(*([i_ori_idx_batch] + [range(batch_size)])))]

        u_i_len_tensor = torch.LongTensor(u_i_len_batch)
        u_ori_idx_tensor = torch.LongTensor(u_ori_idx_batch)
        user_tensor = torch.LongTensor(user_batch)
        u_i_tensor = self._padding_seq(batch_size, u_max_len, u_i_batch)
        u_n_tensor, u_h_adj_tensor, u_r_adj_tensor, u_t_adj_tensor, u_n_len_tensor, u_r_len_tensor = self._padding_graph(batch_size, u_n_batch)

        i_u_len_tensor = torch.LongTensor(i_u_len_batch)
        i_ori_idx_tensor = torch.LongTensor(i_ori_idx_batch)
        item_tensor = torch.LongTensor(item_batch)
        i_u_tensor = self._padding_seq(batch_size, i_max_len, i_u_batch)
        i_n_tensor, i_h_adj_tensor, i_r_adj_tensor, i_t_adj_tensor, i_n_len_tensor, i_r_len_tensor = self._padding_graph(batch_size, i_n_batch)

        label_tensor = torch.LongTensor(label_batch)

        return u_i_len_tensor, u_ori_idx_tensor, user_tensor, u_i_tensor, u_n_tensor, u_h_adj_tensor, u_r_adj_tensor, u_t_adj_tensor, u_n_len_tensor, u_r_len_tensor, \
               i_u_len_tensor, i_ori_idx_tensor, item_tensor, i_u_tensor, i_n_tensor, i_h_adj_tensor, i_r_adj_tensor, i_t_adj_tensor, i_n_len_tensor, i_r_len_tensor, label_tensor

    def _padding_graph(self, batch_size, n_batch):
        n_len_list = [tup[1] for tup in n_batch]
        r_len_list = [tup[3] for tup in n_batch]
        max_n_len = max(n_len_list)
        max_r_len = max(r_len_list)

        n_batch_array = np.zeros([batch_size, max_n_len], dtype=int)

        h_adj_array = np.zeros([batch_size, max_r_len, max_n_len, max_n_len], dtype=int)
        r_adj_array = np.zeros([batch_size, max_r_len, max_n_len, max_n_len], dtype=int)
        t_adj_array = np.zeros([batch_size, max_r_len, max_n_len, max_n_len], dtype=int)

        for i in range(batch_size):
            all_nodes, node_size, rel_adj_dict, _ = n_batch[i]
            n_batch_array[i, :node_size] = all_nodes

            for j, (_, (h_array, r_array, t_array)) in enumerate(rel_adj_dict.items()):
                h_adj_array[i, j, :node_size, :node_size] = h_array
                r_adj_array[i, j, :node_size, :node_size] = r_array
                t_adj_array[i, j, :node_size, :node_size] = t_array

        n_batch_tensor = torch.from_numpy(n_batch_array)
        h_adj_tensor = torch.from_numpy(h_adj_array)
        r_adj_tensor = torch.from_numpy(r_adj_array)
        t_adj_tensor = torch.from_numpy(t_adj_array)
        n_len_tensor = torch.LongTensor(n_len_list)
        r_len_tensor = torch.LongTensor(r_len_list)

        return n_batch_tensor, h_adj_tensor, r_adj_tensor, t_adj_tensor, n_len_tensor, r_len_tensor

    def _padding_seq(self, batch_size, max_len, sorted_seq):
        array = np.zeros([batch_size, max_len], dtype=int)
        for i, s in enumerate(sorted_seq):
            array[i, :len(s)] = np.array(s)

        return torch.from_numpy(array).long()

    def _sorted_pack(self, batch_size, user_batch, u_i_batch, u_t_batch):
        lens = [len(temp) for temp in u_i_batch]
        unsorted_pack = [lens] + [range(batch_size)] + [user_batch] + [u_i_batch] + [u_t_batch]
        sorted_pack = [list(t) for t in zip(*sorted(zip(*unsorted_pack), reverse=True))]
        max_len = max(lens)
        len_batch = sorted_pack[0]
        ori_idx = sorted_pack[1]
        user_batch = sorted_pack[2]
        u_i_batch = sorted_pack[3]
        u_t_batch = sorted_pack[4]
        return max_len, len_batch, ori_idx, user_batch, u_i_batch, u_t_batch

    def _load_rating(self):
        rating_file = '../data/' + self.dataset + '/ratings_final'
        if os.path.exists(rating_file + '.npy'):
            rating_np = np.load(rating_file + '.npy')
        else:
            rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
            np.save(rating_file + '.npy', rating_np)

        rating_np[:, 0] = rating_np[:, 0] + self.OFFSET
        rating_np[:, 1] = rating_np[:, 1] + self.OFFSET

        n_user = max(rating_np[:, 0]) + 1
        n_item = max(rating_np[:, 1]) + 1

        return rating_np, n_user, n_item

    def _split_data(self):
        eval_ratio = 0.2
        test_ratio = 0.2

        n_ratings = self.rating_np.shape[0]

        eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
        left = set(range(n_ratings)) - set(eval_indices)
        test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
        train_indices = list(left - set(test_indices))

        user_items_dict = collections.defaultdict(list)
        ori_rating_dict = collections.defaultdict(list)
        for i in train_indices:
            user = self.rating_np[i][0]
            item = self.rating_np[i][1]
            rating = self.rating_np[i][2]
            ori_rating = self.rating_np[i][3]
            if rating == 1:
                user_items_dict[user].append(item)
                ori_rating_dict[user].append(ori_rating)

        self.sorted_user_items_rating_dict = collections.defaultdict(list)
        sorted_user_items_dict = collections.defaultdict(list)
        for user, items in user_items_dict.items():
            ori_ratings = ori_rating_dict[user]
            unsorted = [ori_ratings] + [range(0, len(items))]
            sort = [list(t) for t in zip(*sorted(zip(*unsorted)))]
            sorted_ratings = sort[0]
            sorted_items = list(np.array(items)[sort[1]])

            sorted_user_items_dict[user] = sorted_items
            self.sorted_user_items_rating_dict[user] = sorted_ratings

        item_user_rating_dict = dict()
        for user, items in sorted_user_items_dict.items():
            for i, item in enumerate(items):
                item_user_rating_dict['i{}=u{}'.format(item, user)] = self.sorted_user_items_rating_dict[user][i]

        train_indices = [i for i in train_indices if self.rating_np[i][0] in user_items_dict]
        eval_indices = [i for i in eval_indices if self.rating_np[i][0] in user_items_dict]
        test_indices = [i for i in test_indices if self.rating_np[i][0] in user_items_dict]

        train_data = self.rating_np[train_indices]
        eval_data = self.rating_np[eval_indices]
        test_data = self.rating_np[test_indices]

        return np.array(train_data), np.array(eval_data), np.array(test_data), sorted_user_items_dict, item_user_rating_dict

    def _load_kg(self):
        kg_file = '../data/' + self.dataset + '/kg_final'
        if os.path.exists(kg_file + '.npy'):
            kg_np = np.load(kg_file + '.npy')
        else:
            kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
            np.save(kg_file + '.npy', kg_np)

        kg_np = kg_np + self.OFFSET

        n_entity = max([max(kg_np[:, 0]), max(kg_np[:, 2])]) + 1
        n_relation = max(kg_np[:, 1]) + 1

        kg = collections.defaultdict(list)
        for head, relation, tail in kg_np:
            kg[head].append((relation, tail))

        return kg, n_entity, n_relation
