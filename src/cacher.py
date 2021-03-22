import h5py

class Cacher:
    def __init__(self, path, point_number_groups, point_size, datasets={}):
        self.point_number_groups = point_number_groups
        self.f = h5py.File(path, 'w', libver='latest')
        self.f.create_dataset('point_nums', data=self.point_number_groups)

        self.pgs = [self.f.create_group(f'point_group{i}-{j}') for i, j in enumerate(self.point_number_groups)]
        for pg, n in zip(self.pgs, point_number_groups):
            chunk_size = point_number_groups[-1]//n
            pg.create_dataset('pcs', (0, n, point_size), maxshape=(None, n, point_size), chunks=(chunk_size, n, point_size))
            for name, size in datasets:
                pg.create_dataset(name, (0, *size), maxshape=(None, *size), chunks=(chunk_size, *size))

    def add_to_group(self, g, data):
        for key in data:
            N = self.pgs[g][key].shape[0]
            self.pgs[g][key].resize(N+1, axis=0)
            self.pgs[g][key][N] = data[key]

    def print_sizes(self):
        print("Sizes of point groups")
        for num_pt, pg in zip(self.point_number_groups, self.pgs):
            print(f'{num_pt}: {pg["pcs"].shape[0]}')

    def close(self):
        self.f.close()
