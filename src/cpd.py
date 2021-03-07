import matplotlib.pyplot as plt
import mesh_sampler
import open3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pyquaternion import Quaternion
from icecream import ic

def l22(X, Y):
    # return (N, M)
    N = X.shape[0]
    M = Y.shape[0]
    return ((X.reshape(N, 1, -1) - Y.reshape(1, M, -1))**2).sum(-1)

def point_dist(X, Y, featX, featY, w):
    if featX is not None and featY is not None:
        #  ic(l22(X, Y))
        #  ic(w * l22(featX, featY))
        return l22(X, Y) + w * l22(featX, featY)
    else:
        return l22(X, Y)

fast_config = {
    "w": 0.0,
    "weight": 0.005,
    "decay": 0.7,
    "noise_decay": 0.5,
    "noise_sigma": 1,
    "size_threshold": 0.05,
    "min_iter": 20,
    "max_iter": 80,
}

symmetric_config = {
    "w": 0.5,
    "weight": 0.005,
    "decay": 0.7,
    "noise_decay": 0.8,
    "noise_sigma": 2,
    "size_threshold": 0.05,
    "min_iter": 20,
    "max_iter": 200,
}

color_config = {
    "w": 0.0,
    "weight": 0.01,
    "decay": 0.8,
    "noise_decay": 0.8,
    "noise_sigma": 2,
    "size_threshold": 0.05,
    "min_iter": 20,
    "max_iter": 200,
}

class CPD:
    def __init__(self, X, Y, config, featX=None, featY=None):
        # X, Y
        # Find X = RY + t
        # (N, E) dim
        self.featX = featX
        self.featY = featY
        # (N, D) dim
        self.X = X
        self.Y = Y
        self.config = config
        self.enable_scale = True

        self.N, self.D = X.shape
        self.M = Y.shape[0]
        self.R = np.eye(self.D)
        self.s = 0.1
        self.t = np.zeros((self.D, 1))
        # Initialize
        self.var = 1/self.D/self.N/self.M * point_dist(self.X, self.Y, self.featX, self.featY, self.config["weight"]).sum()

        #  P = self.random_probs()
        #  self.maximization_step(P)

    def random_probs(self):
        soft_probs = np.random.rand(self.N, self.M) + self.weight * l22(self.featX, self.featY)
        return (soft_probs / soft_probs.sum(1, keepdims=True)).T

    def visualize_points(self):
        expected = (self.s*self.Y @ self.R.T + self.t.T)
        pc2 = open3d.geometry.PointCloud()
        pc2.points = open3d.utility.Vector3dVector(expected)
        if self.featY is not None:
            pc2.colors = open3d.utility.Vector3dVector(self.featY)
        else:
            pc2.paint_uniform_color(np.array([0, 0, 1]).reshape(3, 1))
        pc2.paint_uniform_color(np.array([0, 0, 1]).reshape(3, 1))

        pc1 = open3d.geometry.PointCloud()
        pc1.points = open3d.utility.Vector3dVector(self.X)
        if self.featX is not None:
            pc1.colors = open3d.utility.Vector3dVector(self.featX)
        else:
            pc1.paint_uniform_color(np.array([1, 0, 0]).reshape(3, 1))
        pc1.paint_uniform_color(np.array([1, 0, 0]).reshape(3, 1))
        open3d.visualization.draw_geometries([pc1, pc2])

    def expectation_step(self, weight, sigma):
        expected = (self.s*self.Y @ self.R.T + self.t.T)
        sq_diff = point_dist(self.X, expected, self.featX, self.featY, weight)
        #  noise = np.random.rand(self.N, self.M) * sigma
        #  sq_diff += noise
        soft_probs = np.exp(-sq_diff / 2 / self.var)
        c = self.M/self.N * self.config["w"] / (1-self.config["w"]) * (2*np.pi*self.var)**(self.D/2)
        # Perform soft max comparison across Y to find approx one correspondence per X
        # as a result, we want the target to be X because it might not have a point for each point in Y
        return (soft_probs / (soft_probs.sum(1, keepdims=True) + c)).T

    def maximization_step(self, P):
        N_P = P.sum()
        # (d, N) * (N, M) mean = (d, 1)
        mu_x = (self.X.T @ P.T).sum(1, keepdims=True)/N_P
        mu_y = (self.Y.T @ P).sum(1, keepdims=True)/N_P
        # (n, d)
        Xhat = self.X - mu_x.T
        Yhat = self.Y - mu_y.T
        A = Xhat.T @ P.T @ Yhat
        U, SS, vh = np.linalg.svd(A)
        V = vh.T
        C = np.eye(self.D)
        C[-1, -1] = np.linalg.det(U @ V.T)
        self.R = U @ C @ V.T
        if self.enable_scale:
            self.s = np.trace(A.T @ self.R) / np.trace((Yhat.T * P.sum(1)) @ Yhat)
        else:
            self.s = 1
        self.t = mu_x - self.s*self.R @ mu_y
        self.var = 1 / N_P / self.D * (np.trace(Xhat.T @ np.diag(P.sum(0)) @ Xhat) - self.s*np.trace(A.T @ self.R))

    def calculate_loss(self, P):
        N_P = P.sum()
        # (d, N) * (N, M) mean = (d, 1)
        mu_x = (self.X.T @ P.T).sum(1, keepdims=True)/N_P
        mu_y = (self.Y.T @ P).sum(1, keepdims=True)/N_P
        # (n, d)
        Xhat = self.X - mu_x.T
        Yhat = self.Y - mu_y.T

        p1 = np.trace(Xhat.T @ np.diag(P.T.sum(1)) @ Xhat)
        p2 = 2*self.s*np.trace(Xhat.T @ P.T @ Yhat @ self.R.T)
        p3 = self.s**2*np.trace(Yhat.T @ np.diag(P.sum(1)) @ Yhat)
        C = N_P*self.D/2*np.log(self.var)
        unnorm = 1 / 2 / self.var * (p1+p2+p3) + C
        return unnorm

    def estimate(self, viz=False):
        for i in range(self.config["max_iter"]):
            if viz:
                self.visualize_points()
            weight = self.config["weight"] * self.config["decay"]**i
            noise_sigma = self.config["noise_sigma"] * self.config["noise_decay"]**i
            P = self.expectation_step(weight, noise_sigma)
            if P.sum() == 0 or np.isnan(P.sum()):
                break
            self.maximization_step(P)
            if self.var < 1e-6:
                break
            # Using our knowledge of the true scale
            if abs(self.s - 1) < self.config["size_threshold"]:
                break
        loss = self.calculate_loss(P)
        return loss

    def refine(self, viz=False):
        self.enable_scale = False
        for i in range(20):
            if viz:
                self.visualize_points()
            P = self.expectation_step(0, 0)
            if P.sum() == 0 or np.isnan(P.sum()):
                break
            self.maximization_step(P)
            if self.var < 1e-6:
                break
        loss = self.calculate_loss(P)
        return loss

    def get_transform(self):
        RT = np.eye(4)
        RT[:3, :3] = self.R
        RT[:3, 3] = self.t.squeeze()
        return RT

    def init_random(self):
        # Rand rot
        q = Quaternion(np.random.rand(4))
        q /= q.norm
        self.R = q.rotation_matrix

if __name__ == "__main__":
    import random
    from pathlib import Path
    base_path = Path("/data/haosu")
    meshes = mesh_sampler.MeshSampler(base_path, base_path / "training_data/objects_v1.csv")
    meshes.load_cache()

    key = list(meshes.clouds.keys())[1]
    sample = meshes.clouds[key]
    #  sample = mesh_sampler.iterative_furthest_point_sampling(sample, 10000)

    # Rand rot
    q = Quaternion(np.random.rand(4))
    q /= q.norm
    rt_gt = q.transformation_matrix
    rt_gt[:3, 3] = np.random.rand(3)*0.1

    # Rand ind
    inds = list(range(sample.shape[0]))
    random.shuffle(inds)

    # RUN Full ICP Test
    sample2 = open3d.geometry.PointCloud()
    sample2.points = open3d.utility.Vector3dVector(sample)
    #  sample2.estimate_normals()

    sample1 = open3d.geometry.PointCloud()
    sample1.points = open3d.utility.Vector3dVector(sample[inds])
    #  sample1.normals = open3d.utility.Vector3dVector(np.asarray(sample2.normals)[inds])
    sample2 = sample2.transform(rt_gt)

    pc1 = np.asarray(sample1.points)
    pc2 = np.asarray(sample2.points)
    algo = CPD(pc2, pc1, w=0.2)
    RT = algo.estimate()
    print(RT, rt_gt)
    algo.visualize_points()
