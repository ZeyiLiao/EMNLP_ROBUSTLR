from helper import *


class DataModule(pl.LightningDataModule):

	def __init__(self, dataset, train_dataset, dev_dataset, test_dataset, arch, train_batch_size=32, eval_batch_size=32,\
					num_workers=10, pad_idx=0, ood_test_dataset='', part_of_train=1, filter_unk=False, filter_false=False):
		super().__init__()
		self.p                  = types.SimpleNamespace()
		self.p.dataset          = dataset
		self.p.train_dataset    = train_dataset		# used in load_dataset()
		self.p.dev_dataset      = dev_dataset		# used in load_dataset()
		self.p.test_dataset     = test_dataset		# used in load_dataset()
		self.p.ood_test_dataset = ood_test_dataset
		self.p.actual_arch = arch

		if arch in ["t5_11b","t5_3b"]:
			arch = "t5_large"

		self.p.arch             = arch
		self.p.train_batch_size = train_batch_size
		self.p.eval_batch_size  = eval_batch_size
		self.p.num_workers      = num_workers
		self.p.pad_idx          = pad_idx
		self.p.part_of_train    = part_of_train
		self.p.filter_unk       = filter_unk
		self.p.filter_false     = filter_false

	def load_dataset(self, split, arch):
		dataset = ddict(list)

		if arch.startswith('t5'):
			all_folders = [f'../data/processed/{x}/t5_large/{split}/' for x in getattr(self.p, f'{split}_dataset').split(',')]
			print(f'allfolders for {split}: {all_folders}')
			for key in ['input_ids', 'output_ids','has_neg', 'theory_id']:
				for folder in all_folders:
					with open(folder + f'{key}.pkl', 'rb') as f:
						try:
							with open(folder + f'{key}.pkl', 'rb') as f:
								tmp          = pickle.load(f)
								dataset[key] = dataset[key] + tmp
						except Exception as e:
							print(f'Missing Key {key}')
							assert key == 'theory_id'

		else:
			all_folders = [f'../data/processed/{x}/{arch}/{split}/' for x in getattr(self.p, f'{split}_dataset').split(',')]
			print(f'allfolders for {split}: {all_folders}')
			for key in ['input_ids', 'label', 'ctx_ids', 'stmt_ids', 'has_neg', 'theory_id']:
				for folder in all_folders:
					try:
						with open(folder + f'{key}.pkl', 'rb') as f:
							tmp          = pickle.load(f)
							dataset[key] = dataset[key] + tmp
					except Exception as e:
						print(f'Missing Key {key}')
						assert key == 'theory_id'

		dataset = dict(dataset)
		if split == 'train':
			if self.p.arch == 't5_large':
				keys = ['input_ids', 'output_ids','has_neg', 'theory_id']
			elif self.p.arch == 'roberta_large_race' or self.p.arch == 'albert_xxlarge_v2':
				keys = ['input_ids', 'label', 'ctx_ids', 'stmt_ids', 'has_neg', 'theory_id']

			part_of_train_num = int(len(dataset[key]) * self.p.part_of_train)
			for key in keys:
				dataset[key] = dataset[key][:part_of_train_num]
				print(f"Only use {self.p.part_of_train} * total length for train")
		return dataset

	def load_ood_dataset(self, arch):
		dataset = ddict(list)
		if arch.startswith('t5'):
			all_folders = [f'../data/processed/{self.p.ood_test_dataset}/t5_large/test/']
			print('OOD folders', all_folders)
			for key in ['input_ids', 'output_ids','has_neg', 'theory_id']:
				for folder in all_folders:
					with open(folder + f'{key}.pkl', 'rb') as f:
						try:
							with open(folder + f'{key}.pkl', 'rb') as f:
								tmp          = pickle.load(f)
								dataset[key] = dataset[key] + tmp
						except Exception as e:
							print(f'Missing Key {key}')
							assert key == 'theory_id'

		elif arch == 'roberta_large_race' or arch == 'albert_xxlarge_v2':
			all_folders = [f'../data/processed/{self.p.ood_test_dataset}/{arch}/test/']
			print('OOD folders', all_folders)
			for key in ['input_ids', 'label', 'ctx_ids', 'stmt_ids', 'has_neg', 'theory_id']:
				for folder in all_folders:
					try:
						with open(folder + f'{key}.pkl', 'rb') as f:
							tmp          = pickle.load(f)
							dataset[key] = dataset[key] + tmp
					except Exception as e:
						print(f'Missing Key {key}')
						assert key == 'theory_id'

		return dict(dataset)

	def load_classifier_dataset(self, split):
		dataset = []
		all_folders = [f'../data/processed/{self.p.dataset}/classifier/']
		print('Folders', all_folders)
		for folder in all_folders:
			with open(f'{folder}/{split}.csv', 'r') as f:
				reader = csv.reader(f)
				for row in reader:
					dataset.append([int(x) for x in row])

		return dataset

	def setup(self, stage=None, splits='all'):
		self.data = ddict(list)
		if splits == 'all':
			splits = ['train', 'dev', 'test']

		for split in splits:
			if self.p.arch == 't5_large' or self.p.arch == 't5_11b':
				self.data[split] = GenerativeDataset(self.load_dataset(split, self.p.arch), self.p.pad_idx, filter_unk=self.p.filter_unk, filter_false=self.p.filter_false)
			elif self.p.arch == 'roberta_large_race' or self.p.arch == 'albert_xxlarge_v2':
				self.data[split] = DiscriminativeDataset(self.load_dataset(split, self.p.arch), self.p.pad_idx, filter_unk=self.p.filter_unk, filter_false=self.p.filter_false)
			elif self.p.arch == 'linear':
				self.data[split] = ClassifierDataset(self.load_classifier_dataset(split))

		if self.p.ood_test_dataset != '':
			if self.p.arch == 't5_large' or self.p.arch == 't5_11b':
				self.data['ood_test'] = GenerativeDataset(self.load_ood_dataset(self.p.arch), self.p.pad_idx, iid=False, filter_unk=self.p.filter_unk, filter_false=self.p.filter_false)
			elif self.p.arch == 'roberta_large_race' or self.p.arch == 'albert_xxlarge_v2':
				self.data['ood_test'] = DiscriminativeDataset(self.load_ood_dataset(self.p.arch), self.p.pad_idx, iid=False, filter_unk=self.p.filter_unk, filter_false=self.p.filter_false)

	def train_dataloader(self, shuffle=True):
		return DataLoader(
					self.data['train'],
					batch_size=self.p.train_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['train'].collater,
					shuffle=shuffle,
					pin_memory=True
				)

	def val_dataloader(self):
		return DataLoader(
					self.data['dev'],
					batch_size=self.p.eval_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['dev'].collater,
					pin_memory=True
				)

	def test_dataloader(self, split='test'):
		if self.p.actual_arch in ["t5_11b","t5_3b"]:

			self.p.eval_batch_size = 32
			self.p.num_workers = 10
			print(f"For {self.p.actual_arch} we scale up the test batch size otherwise it will cost so much time \n")
			print(f"We scale up the batch size to {self.p.eval_batch_size}, num workers to {self.p.num_workers}")

		return DataLoader(
					self.data[split],
					batch_size=self.p.eval_batch_size,
					num_workers=self.p.num_workers,
					collate_fn=self.data['test'].collater,
					pin_memory=True
				)

	@staticmethod
	def add_data_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument("--dataset", 		 				    type=str)
		parser.add_argument("--train_dataset",	    default='', 	type=str)
		parser.add_argument("--dev_dataset",	    default='', 	type=str)
		parser.add_argument("--test_dataset",	    default='', 	type=str)
		parser.add_argument("--ood_test_dataset",	default='', 	type=str)
		parser.add_argument("--num_workers", 	    default=10, 	type=int)
		return parser


class DiscriminativeDataset(Dataset):

	def __init__(self, dataset, pad_idx, iid=True, filter_unk=False, filter_false=False):
		self.data    = filter_label(dataset, 'discriminative', filter_unk, filter_false)
		self.pad_idx = pad_idx
		self.iid     = iid
		self.filter_false = filter_false

	def __len__(self):
		return len(self.data['label'])

	def __getitem__(self, idx):
		theory_id = self.data['theory_id'][idx] if 'theory_id' in self.data else 'None'
		lbl = (self.data['label'][idx]-1) if self.filter_false else self.data['label'][idx]

		item = {
			'sent'     : torch.LongTensor(self.data['input_ids'][idx]),
			'ctx'      : torch.LongTensor(self.data['ctx_ids'][idx]),
			'stmt'     : torch.LongTensor(self.data['stmt_ids'][idx]),
			'lbl'      : torch.LongTensor([lbl]),
			'ctx_len'  : torch.LongTensor([len(self.data['ctx_ids'][idx])]),
			'stmt_len' : torch.LongTensor([len(self.data['stmt_ids'][idx])]),
			'has_neg'  : torch.LongTensor([1 if self.data['has_neg'][idx] == 'True' else 0]),
			'iid'      : self.iid,
			'theory_id': theory_id,
		}

		return item

	def collater(self, items):
		all_sents = pad_sequence([x['sent'] for x in items], batch_first=True, padding_value=self.pad_idx)

		batch = {
			'all_sents' : all_sents,
			'all_ctxs'  : pad_sequence([x['ctx'] for x in items], batch_first=True, padding_value=self.pad_idx),
			'all_stmts' : pad_sequence([x['stmt'] for x in items], batch_first=True, padding_value=self.pad_idx),
			'all_lbls'  : torch.cat([x['lbl'] for x in items]),
			'ctx_lens'  : torch.cat([x['ctx_len'] for x in items]),
			'stmt_lens' : torch.cat([x['stmt_len'] for x in items]),
			'attn_mask' : (all_sents != self.pad_idx).long(),
			'has_neg'   : torch.cat([x['has_neg'] for x in items]),
			'iid'       : [x['iid'] for x in items],
			'theory_ids': [x['theory_id'] for x in items],
		}

		return batch


class GenerativeDataset(Dataset):

	def __init__(self, dataset, pad_idx, iid=True, filter_unk=False, filter_false=False):
		self.data    = dataset
		self.pad_idx = pad_idx
		self.iid     = iid

	def __len__(self):
		return len(self.data['input_ids'])

	def __getitem__(self, idx):
		theory_id = self.data['theory_id'][idx] if 'theory_id' in self.data else 'None'

		item = {
			'input'    : torch.LongTensor(self.data['input_ids'][idx]),
			'output'   : torch.LongTensor(self.data['output_ids'][idx]),
			'has_neg'  : torch.LongTensor([1 if self.data['has_neg'][idx] == 'True' else 0]),
			'iid'      : self.iid,
			'theory_id': theory_id,
		}

		return item

	def collater(self, items):
		all_inps        = pad_sequence([x['input'] for x in items], batch_first=True, padding_value=self.pad_idx)
		all_outs        = pad_sequence([x['output'] for x in items], batch_first=True, padding_value=self.pad_idx)

		labels = all_outs.clone()
		labels[labels == self.pad_idx] = -100

		batch = {
			'all_inps'         : all_inps,
			'attn_mask'        : (all_inps != self.pad_idx).long(),
			'labels'           : labels,
			'labels_for_decode': all_outs,
			'has_neg'          : torch.cat([x['has_neg'] for x in items]),
			'iid'              : [x['iid'] for x in items],
			'theory_ids'       : [x['theory_id'] for x in items],
		}

		return batch


class ClassifierDataset(Dataset):

	def __init__(self, dataset):
		self.data = dataset

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):

		item = {
			'inp' : torch.FloatTensor(self.data[idx][:-2]),
			'lbl' : torch.LongTensor([self.data[idx][-2]]),
		}

		return item

	def collater(self, items):
		batch = {
			'all_lbls' : torch.cat([x['lbl'] for x in items]),
			'all_inps' : torch.stack([x['inp'] for x in items]),
		}

		return batch


def filter_label(dataset, data_type, filter_unk=False, filter_false=False):
	if filter_unk:
		lbl_to_filter = 2
	elif filter_false:
		lbl_to_filter = 0
	else:
		return dataset

	# returns a filtered version of the dataset with the label 2 removed (corresponding to UNK)
	if data_type == 'discriminative':
		labels = np.array(dataset['label'])
		label_mask = (labels != lbl_to_filter).tolist()

		# filter everything using the mask
		for k,v in dataset.items():
			dataset[k] = list(itertools.compress(v, label_mask))
	elif data_type == 'generative':
		return dataset

	return dataset
