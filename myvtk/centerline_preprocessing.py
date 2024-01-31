import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.stats import zscore
import glob
from myvtk.GetMakeVtk import GetMyVtk, makeVtkFile, measure_length
import pandas as pd
from scipy.spatial.transform import Rotation as R
from procrustes import orthogonal
from myvtk.General import mkdir
from datetime import datetime
import geomstats.geometry.pre_shape as pre_shape
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import geomstats.geometry.discrete_curves as dc
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import HypersphereMetric
from scipy.spatial import distance
import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.discrete_curves import DiscreteCurves
from geomstats.learning.frechet_mean import FrechetMean
import numpy as np
from scipy.interpolate import interp1d




def compute_frechet_mean(curves):
    r2 = Euclidean(dim=3)
    k_sampling_points = 64  # Assuming the sampling points match the size of your curves

    curves_r2 = DiscreteCurves(ambient_manifold=r2, k_sampling_points=k_sampling_points)

    # Compute the Fréchet mean
    frechet_mean = FrechetMean(metric=curves_r2.metric, max_iter=100)
    frechet_mean.fit(curves)

    # Return the mean shape
    return frechet_mean.estimate_

# Example usage:
# curves = np.load('curves.npy')
# mean_shape = compute_frechet_mean(curves)


def align_procrustes(curves,base_id=0,external_curve=None):
    if base_id>-1:
        base_curve = curves[base_id]
    else:
        base_curve = external_curve
    new_curves = []
    for i in range(len(curves)):
        curve = curves[i]
        # 对齐当前曲线到基准曲线
        result = orthogonal(base_curve, curve, translate=True, scale=False) # orthogonal：旋转曲线找点之间距离最小的那个，作为对齐结束后的曲线
        # Z是Procrustes变换后的曲线，用它来更新原始曲线
        # curves[i] = Z
        new_curves.append(result['new_b'])
    return np.array(new_curves)

def arc_length_parametrize(curve):
    # 计算曲线上每两个连续点之间的距离
    diffs = np.diff(curve, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))

    # 累加这些距离得到每个点的弧长
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

    # 生成新的参数值（等间距）
    num_points = len(curve)
    new_params = np.linspace(0, arc_lengths[-1], num_points)

    # 对曲线的每一个维度进行插值
    new_curve = np.zeros_like(curve)
    for i in range(3):
        interp = interp1d(arc_lengths, curve[:, i], kind='cubic')
        new_curve[:, i] = interp(new_params)
    return new_curve

def calculate_srvf(curve):
    # 计算曲线的导数
    t = np.linspace(0, 1, curve.shape[0])
    cs = CubicSpline(t, curve)
    derivative = cs(t, 1)
    
    # 计算SRVF
    magnitude = np.linalg.norm(derivative, axis=1)
    eps = 1e-8  # add a small positive number to avoid division by zero
    srvf = np.sqrt(magnitude + eps)[:, np.newaxis] * derivative / (magnitude[:, np.newaxis] + eps)
    return srvf

def inverse_srvf(srvf, initial_point):
    # print ("calling inverse_srvf, magnitude = np.linalg.norm(srvf, axis=1)")
    # 计算速度
    # magnitude = np.linalg.norm(srvf, axis=1)
    magnitude = np.linalg.norm(srvf, axis=1) ** 2
    velocity = srvf / np.sqrt(magnitude[:, np.newaxis])

    # 对速度进行积分
    curve = np.cumsum(velocity, axis=0)
    
    # 将曲线的初始点设为给定的初始点
    curve = curve - curve[0] + initial_point

    return curve


def align_icp(curves,base_id=80,external_curve=None):
    # 使用第一条曲线作为基准曲线
    new_curves = []
    if base_id > -1:
        base_curve = curves[base_id]
    elif base_id == -1:
        base_curve = np.mean(curves, axis=0)
    elif base_id == -2:
        base_curve = external_curve
    
    for i in range(len(curves)):
        curve = curves[i]
        
        # 计算基准曲线和当前曲线的质心
        base_centroid = np.mean(base_curve, axis=0)
        curve_centroid = np.mean(curve, axis=0)
        
        # 将两个曲线移到原点
        base_centered = base_curve - base_centroid
        curve_centered = curve - curve_centroid
        
        # 计算最优旋转
        H = np.dot(base_centered.T, curve_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
           Vt[-1,:] *= -1
           R = np.dot(Vt.T, U.T)
        
        # 旋转和平移当前曲线以对齐到基准曲线
        # curves[i] = np.dot((curve - curve_centroid), R.T) + base_centroid
        new_curves.append(np.dot((curve - curve_centroid), R.T) + base_centroid)

    return np.array(new_curves)


def compute_geodesic_dist_between_two_curves(curve_A, curve_B):
    # 创建离散曲线对象
    discrete_curves_space = dc.DiscreteCurves(ambient_manifold=dc.Euclidean(dim=3))
    # 确保曲线维度正确
    if curve_A.shape != curve_B.shape:
        raise ValueError("The dimensions of the two curves should be the same.")
    # 计算测地线距离
    geodesic_distance = discrete_curves_space.metric.dist(curve_A, curve_B)
    return geodesic_distance


def compute_geodesic_dist(curves, external=False, external_curve=None):
    geodesic_d = []
    if external == False:
        curve_A = np.mean(curves, axis=0)
    else:
        curve_A = external_curve
    for idx in range(len(curves)):
        curve_B = curves[idx]
        # 创建离散曲线对象
        discrete_curves_space = dc.DiscreteCurves(ambient_manifold=dc.Euclidean(dim=3))
        # 计算测地线距离
        print (curve_A.shape, curve_B.shape)
        geodesic_distance = discrete_curves_space.metric.dist(curve_A, curve_B)
        geodesic_d.append(geodesic_distance)
    return np.array(geodesic_d)

def recovered_curves(inverse_data, is_srvf):
    recovered_curves = []
    if is_srvf:
        for cv in inverse_data:
            recovered_curves.append(inverse_srvf(cv, np.zeros(3)))
    else:
        recovered_curves = inverse_data
    return np.array(recovered_curves)

def write_curves_to_vtk(curves, files, dir):
    for i in range(len(curves)):
        filename = files[i].split("\\")[-1].split(".")[0]
        makeVtkFile(dir+"{}.vtk".format(filename), curves[i], [],[])

def process_data_key(data_key, train_inverse, test_inverse, train_files, test_files, inverse_data_dir):
    is_srvf = "SRVF" in data_key
    train_inverse = recovered_curves(train_inverse, is_srvf)
    test_inverse = recovered_curves(test_inverse, is_srvf)

    inverse_dir = mkdir(inverse_data_dir, "coords_" + data_key)
    write_curves_to_vtk(train_inverse, train_files, inverse_dir + "train_")
    write_curves_to_vtk(test_inverse, test_files, inverse_dir + "test_")

    # print (train_inverse.shape, test_inverse.shape)


def plot_curves_with_arrows(coord1, coord2, Procrustes_curves, Procs_srvf_curves, Typevalues, save_path):
    fig, axs = plt.subplots(2, 2, figsize=(7, 10), dpi=300)
    labels = ['C', 'U', 'S', 'V']

    for index, label in enumerate(labels):
        ax = axs.flatten()[index]

        for i, type_value in enumerate(Typevalues):
            if type_value == label:
                # 绘制黑色的Procrustes_curves曲线
                ax.plot(Procrustes_curves[i][:, coord1], Procrustes_curves[i][:, coord2], '-o', color='black')

                # 在每个点上绘制红色箭头
                for j in range(64):
                    start_point = [Procrustes_curves[i][j, coord1], Procrustes_curves[i][j, coord2]]
                    end_point_delta = [Procs_srvf_curves[i][j, coord1], Procs_srvf_curves[i][j, coord2]]
                    end_point = [start_point[0] + end_point_delta[0] / 3, start_point[1] + end_point_delta[1] / 3]
                    ax.annotate("", xy=end_point, xytext=start_point,
                                arrowprops=dict(arrowstyle="->", color='red'))

                # 为每种标签只绘制一条曲线，所以找到后立即退出循环
                break 

        ax.set_title(f'Type: {label}')
        ax.grid(True)
        ax.invert_yaxis()  # 这里颠倒y轴

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_weighted_procrustes(Procrustes_curves, Curvatures, landmark_w):
    # 这是个没用上的函数。
    interpolated_curvatures = np.zeros((len(Curvatures), len(Procrustes_curves[0])))
    multiplicative_factors = np.zeros_like(Procrustes_curves)

    for i in range(len(Curvatures)):
        interpolated_curvatures[i] = np.interp(np.linspace(0, 1, len(Procrustes_curves[i])),
                                               np.linspace(0, 1, len(Curvatures[i])),
                                               Curvatures[i])
        multiplicative_factors[i] = Procrustes_curves[i] * interpolated_curvatures[i][:, np.newaxis]

    weighted_procrustes_curves = multiplicative_factors * landmark_w[:, np.newaxis]
    return weighted_procrustes_curves


def compute_geod_dist_mat(weighted_procrustes_curves):
    # 根据weighted_procrustes_curves计算距离矩阵
    geod_dist_mat = np.zeros((len(weighted_procrustes_curves), len(weighted_procrustes_curves)))
    for i in range(len(weighted_procrustes_curves)):
        for j in range(len(weighted_procrustes_curves)):
            geodesic_d = compute_geodesic_dist_between_two_curves(weighted_procrustes_curves[i],
                                                                  weighted_procrustes_curves[j])
            geod_dist_mat[i, j] = geodesic_d
    return geod_dist_mat