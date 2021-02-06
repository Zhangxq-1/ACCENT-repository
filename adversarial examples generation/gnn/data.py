from torch.utils.data import Dataset

import utils


class CodePtrDataset(Dataset):

    def __init__(self, source_path, code_path, nl_path):
        # get lines
        sources = utils.load_dataset(source_path)
        codes = utils.load_dataset(code_path)
        nls = utils.load_dataset(nl_path)
        print(len(sources))
        print(len(nls))
        print(len(codes))

        if len(sources) != len(codes) or len(codes) != len(nls):
            raise Exception('The lengths of three dataset do not match.')

        self.sources, self.codes, self.nls = utils.filter_data(sources, codes, nls)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.sources[index], self.codes[index], self.nls[index]

    def get_dataset(self):
        return self.sources, self.codes, self.nls



class CodePtrDataset_1(Dataset):

    def __init__(self, source, code, nl):
        # get lines
        sources = [source]
        codes = [code]
        nls = [nl]

        if len(sources) != len(codes) or len(codes) != len(nls):
            raise Exception('The lengths of three dataset do not match.')

        self.sources, self.codes, self.nls = utils.filter_data(sources, codes, nls)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.sources[index], self.codes[index], self.nls[index]

    def get_dataset(self):
        return self.sources, self.codes, self.nls

