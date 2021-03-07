import matplotlib.pyplot as plt
import mesh_sampler
import open3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pyquaternion import Quaternion


def compute_distance_threshold(distances, D):
    u = np.mean(distances)
    s = np.std(distances, ddof=1)
    if u < D:
        print("High quality")
        return u + 3*s, 60
    elif u < 3*D:
        print("Medium quality")
        return u + 3*s, 60
    elif u < 6*D:
        print("Low quality")
        return u + s, 60
    else:
        print("Very Low quality")
        # Get the value corresponding the biggest peak
        hist, edges = np.histogram(distances, bins=10)
        peak = np.argmax(hist)
        if peak == len(hist)-1:
            return edges[-1], 60
        else:
            # find the next valley
            for i in range(peak, len(hist)-1):
                if hist[i] < hist[i+1] and hist[i] < 0.6*len(distances):
                    return edges[i], 60
            print("max")
            return edges[-1], 60

class ICP:
    def __init__(self, pcloud1, pcloud2, k=10):
        # the original algorithm does not vary k
        self.pcloud1 = pcloud1
        self.pcloud2 = pcloud2
        self.pcloud1.paint_uniform_color(np.array([1, 0, 0]).reshape(3, 1))
        self.pcloud2.paint_uniform_color(np.array([0, 0, 1]).reshape(3, 1))
        # 2a
        if not self.pcloud1.has_normals():
            self.pcloud1.estimate_normals()
        if not self.pcloud2.has_normals():
            self.pcloud2.estimate_normals()
        # (N, 3)
        self.pc1 = np.asarray(pcloud1.points)
        self.pc2 = np.asarray(pcloud2.points)
        self.nc1 = np.asarray(pcloud1.normals)
        self.nc2 = np.asarray(pcloud2.normals)
        # 2b
        self.nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.pc2)
        # Need to compute D
        distances, inds = self.nbrs.kneighbors(self.pc2, n_neighbors=2)
        distances = [max(ds) for ds in distances]
        #  angles = [np.arccos(self.nc2[i] @ self.nc2[j].T)
        # 1
        self.D = np.mean(distances)
        #  print(self.D)
        self.dist_thres = 40 * self.D
        self.angle_thres = 60
        self.full_RT = np.eye(4)

    def match(self, dist_thres, angle_thres=60):
        # step two: get nearest neighbors from cloud with fewer to more points
        matches = -np.ones(self.pc2.shape[0], dtype=np.int)
        mdists = np.zeros(self.pc2.shape[0])
        mangles = np.zeros(self.pc2.shape[0])
        distances, indices = self.nbrs.kneighbors(self.pc1)
        for i, (dists, inds) in enumerate(zip(distances, indices)):
            # step three: iterate through the neighbors and filter out based on the following criteria:
            #  orientation: the angle between the two normals cannot be bigger than the maximum of the angle expected (60 deg in paper)
            #  distance: the distance cannot be larger than our threshold
            #  color: the color of the two points must be identical TODO
            #  if none of the points fit the criteria, then we don't output a match
            angles = (self.nc1[i].reshape(1, 3) @ self.nc2[inds].T).reshape(-1)
            #  angles = np.arccos(angles) * 180 / np.pi
            remaining = np.where((dists < dist_thres) & (angles > np.cos(angle_thres/180*np.pi)))
            if len(remaining[0]) == 0:
                continue
            dists = dists[remaining]
            angles = angles[remaining]
            inds = inds[remaining]
            j = np.argmax(-dists)
            matches[i] = inds[j]
            mdists[i] = dists[j]
            mangles[i] = angles[j]
        return matches, mdists, mangles

    def draw_matches(self, matches):
        points = open3d.utility.Vector3dVector(np.concatenate([self.pc1, self.pc2], axis=0))
        inds, = np.where(matches != -1)
        i = np.arange(len(matches))[inds]
        j = matches[inds]+self.pc1.shape[0]
        lines = np.stack([i, j], axis=0)
        inds = np.where(lines[:, 1] != -1)
        lines = lines[inds]
        line_geo = open3d.geometry.LineSet(
                points=open3d.utility.Vector3dVector(points),
                lines=open3d.utility.Vector2iVector(lines.T))
        mpc1, mpc2 = self.register(matches)

        pc1 = open3d.geometry.PointCloud()
        pc1.points = open3d.utility.Vector3dVector(mpc1)
        pc1.paint_uniform_color(np.array([1, 0, 0]).reshape(3, 1))
        pc2 = open3d.geometry.PointCloud()
        pc2.points = open3d.utility.Vector3dVector(mpc2)
        pc2.paint_uniform_color(np.array([0, 0, 1]).reshape(3, 1))

        #  open3d.visualization.draw_geometries([pc1, pc2, line_geo])
        open3d.visualization.draw_geometries([self.pcloud1, self.pcloud2, line_geo])

    def register(self, matches):
        inds, = np.where(matches != -1)
        mpc2 = self.pc2[matches[inds], :]
        mpc1 = self.pc1[np.arange(len(matches))[inds], :]
        return mpc1, mpc2

    def single_iter(self):
        # 3a: find matches
        matches, mdists, mangles = self.match(self.dist_thres, self.angle_thres)
        #  self.draw_matches(matches)
        # 3b: update recovered matches
        mpc1, mpc2 = self.register(matches)
        distances = np.linalg.norm(mpc2-mpc1, axis=1)
        self.dist_thres, self.angle_thres = compute_distance_threshold(distances, self.D)
        # update matches
        reject = np.where(mdists > self.dist_thres)
        matches[reject] = -1
        mpc1, mpc2 = self.register(matches)


        # 3c: estimate motion
        RT = estimate_rt(mpc1, mpc2)
        # 4d: apply motion to first frame
        self.apply_rt(RT)
        return RT

    def apply_rt(self, RT):
        pc1 = RT @ np.concatenate((self.pc1.T, np.ones((1, self.pc1.shape[0]))), axis=0)
        self.pc1 = pc1[:3, :].T
        self.pcloud1 = self.pcloud1.transform(RT)
        self.full_RT = RT @ self.full_RT

    def estimate(self, delta_r_thres = 0.5, delta_t_thres=0.005):
        # First, initialize by aligning centers
        RT = np.eye(4)
        RT[:3, 3] = np.mean(self.pc2, axis=0) - np.mean(self.pc1, axis=0)
        self.apply_rt(RT)

        # run iterations
        last_r = np.array([1, 0, 0])
        last_t = np.array([0, 0, 0])
        for i in range(40):
            RT = self.single_iter()
            # determine whether to stop
            t = RT[:3, 3]
            delta_t = np.linalg.norm(t-last_t)
            q = Quaternion(matrix=RT[:3, :3])
            r = q.degrees * q.axis
            delta_r = np.linalg.norm(last_r - r)/np.linalg.norm(r)
            if delta_t < delta_t_thres and delta_r < delta_r_thres:
                print(f"Convergence: {i}")
                break
            last_r = r
            last_t = t
        return self.full_RT


def estimate_rt(mpc1, mpc2, w=None):
    # minimizes \|Rx + t - y\| where x = mpc1, y = mpc2
    # dual quaternion algebra method
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.453.2570&rep=rep1&type=pdf
    def sk(v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]])
    def Q(q):
        # w: scalar part
        # v: imaginary part
        w = q[3]
        v = q[:3]
        ret = np.zeros((4, 4))
        ret[:3, :3] = w * np.eye(3) + sk(v)
        ret[:3, 3] = v
        ret[3, :3] = -v
        ret[3, 3] = w
        return ret

    def W(q):
        w = q[3]
        v = q[:3]
        # w: scalar part
        # v: imaginary part
        ret = np.zeros((4, 4))
        ret[:3, :3] = w * np.eye(3) - sk(v)
        ret[:3, 3] = v
        ret[3, :3] = -v
        ret[3, 3] = w
        return ret

    N = mpc1.shape[0]
    if w is None:
        w = np.ones(N)
    # N, 4
    x = np.concatenate([mpc1, np.zeros((N, 1))], axis=1)
    y = np.concatenate([mpc2, np.zeros((N, 1))], axis=1)

    # Minimize
    C1 = np.zeros((4, 4))
    C2 = np.zeros((4, 4))
    for i in range(N):
        C1 += -2 * w[i] * Q(y[i]).T @ W(x[i])
        C2 += 2 * w[i] * (W(x[i]) - Q(y[i]))
    A = 0.5 * (C2.T @ C2 /2/w.sum() - C1 - C1.T)

    vals, vecs = np.linalg.eig(A)
    q = vecs[:, np.argmax(vals)]

    # Convert quaternion to transform matrix
    RT = np.eye(4)
    qu = Quaternion(w=q[3], x=q[0], y=q[1], z=q[2])
    RT = qu.transformation_matrix
    s = -C2 @ q / 2 / w.sum()
    RT[:3, 3] = (W(q).T@s)[:3]

    return RT

if __name__ == "__main__":
    import random
    from pathlib import Path
    base_path = Path("/data/haosu")
    meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
    meshes.load_cache()

    key = list(meshes.clouds.keys())[1]
    sample = meshes.clouds[key]
    sample = mesh_sampler.iterative_furthest_point_sampling(sample, 100)

    q = Quaternion(np.random.rand(4))
    q /= q.norm
    rt_gt = q.transformation_matrix
    rt_gt[:3, 3] = np.random.rand(3)*0.1

    # RUN Estimate RT Test
    sample2 = open3d.geometry.PointCloud()
    sample2.points = open3d.utility.Vector3dVector(sample)
    sample2.estimate_normals()
    sample2 = sample2.transform(rt_gt)

    sample1 = open3d.geometry.PointCloud()
    sample1.points = open3d.utility.Vector3dVector(sample)
    sample1.normals = sample2.normals
    #  print(sample1, sample2)

    #  open3d.visualization.draw_geometries([sample1, sample2])
    RT = estimate_rt(np.asarray(sample1.points), np.asarray(sample2.points))
    print(f"Estimate RT passed: {np.allclose(RT, rt_gt)}")

    # RUN Simple Matching Test
    inds = list(range(sample.shape[0]))
    random.shuffle(inds)
    sample2 = open3d.geometry.PointCloud()
    sample2.points = open3d.utility.Vector3dVector(sample)
    sample2.estimate_normals()

    sample1 = open3d.geometry.PointCloud()
    sample1.points = open3d.utility.Vector3dVector(sample[inds])
    sample1.normals = open3d.utility.Vector3dVector(np.asarray(sample2.normals)[inds])

    icp = ICP(sample1, sample2)
    #  print(icp.pc1, icp.pc2)
    matches, mdist, mangles = icp.match(0.1)
    print(f"Simple Match Test: {(matches == inds).all()}")
    #  icp.draw_matches(matches)

    # RUN Full ICP Test
    sample2 = open3d.geometry.PointCloud()
    sample2.points = open3d.utility.Vector3dVector(sample)
    sample2.estimate_normals()

    sample1 = open3d.geometry.PointCloud()
    sample1.points = open3d.utility.Vector3dVector(sample[inds])
    sample1.normals = open3d.utility.Vector3dVector(np.asarray(sample2.normals)[inds])
    sample2 = sample2.transform(rt_gt)

    icp = ICP(sample1, sample2)
    RT = icp.estimate()
    print(RT)
    #  sample1 = sample1.transform(RT)

    open3d.visualization.draw_geometries([sample1, sample2])
    print(rt_gt)
    print(RT)
