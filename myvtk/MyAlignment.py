import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA, KernelPCA
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.stats import zscore
import glob
from myvtk.GetMakeVtk import *
import pandas as pd
from scipy.spatial.transform import Rotation as R
from procrustes import orthogonal
from myvtk.General import *
from datetime import datetime
import geomstats.geometry.pre_shape as pre_shape
import geomstats.geometry.discrete_curves as dc
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.hypersphere import HypersphereMetric
from scipy.spatial import distance
from myvtk.centerline_preprocessing import *
from scipy import interpolate
import matplotlib
import matplotlib.cm as cm
from scipy.spatial.distance import euclidean
from myvtk.Mypca import *
import shutil
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter
import matplotlib.gridspec as gridspec
import csv
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal, kde
import seaborn as sns
import copy
import joblib
from myvtk.geometry import *
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from minisom import MiniSom
from sklearn.neighbors import KernelDensity
from myvtk.synthetic import *
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from myvtk.Mymetrics import *
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
import warnings
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import minimize
from myvtk.mygeodesic_plot import *
import platform
from dtw import *
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from collections import defaultdict
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import pearsonr
import networkx as nx  # 导入NetworkX库
from myvtk.MyRiemannCov import *
from scipy.signal import find_peaks
from geomstats.learning.pca import TangentPCA
from geomstats.geometry.discrete_curves import ElasticMetric, SRVMetric
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PIL import Image
from myvtk.Myscores import *
from myvtk.MytangentPCA import *
warnings.filterwarnings("ignore")
import matplotlib as mpl
import dill as pickle
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# def compute_cost_matrix(curve1, curve2):
#     # Compute the cost matrix between two curves
#     n = curve1.shape[0]
#     m = curve2.shape[0]
#     cost_matrix = np.zeros((n, m))
    
#     for i in range(n):
#         for j in range(m):
#             cost_matrix[i, j] = np.linalg.norm(curve1[i] - curve2[j])
#     return cost_matrix

def find_optimal_reparametrization(cost_matrix):
    n, m = cost_matrix.shape
    dp_matrix = np.inf * np.ones((n, m))
    dp_matrix[0, 0] = cost_matrix[0, 0]
    
    for i in range(1, n):
        for j in range(1, m):
            choices = [dp_matrix[i-1, j], dp_matrix[i, j-1], dp_matrix[i-1, j-1]]
            dp_matrix[i, j] = cost_matrix[i, j] + min(choices)
    
    # Backtracking
    i, j = n-1, m-1
    path = [(i, j)]
    while i > 0 and j > 0:
        step = np.argmin([dp_matrix[i-1, j], dp_matrix[i, j-1], dp_matrix[i-1, j-1]])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
        path.append((i, j))
    
    path.reverse()
    return path

def interpolate_pt(curve, num_points):
    # 插值以增加曲线上的点
    # curve 是一个 numpy 数组，形状为 [n, 3]
    # num_points 是插值后曲线上的点的数量
    t = np.linspace(0, 1, len(curve))
    t_new = np.linspace(0, 1, num_points)
    curve_interpolated = np.array([np.interp(t_new, t, dim) for dim in curve.T]).T
    return curve_interpolated

# def reparameterize_curve(curve, warping_function):
#     # 使用 warping function 重排列曲线点
#     # 注意：这里假设 warping_function 已经考虑了插值后的点的数量
#     reparam_curve = curve[warping_function]
#     return reparam_curve

def extract_increasing_mappings(warping_function):
    # 初始化一个字典，用于存储每个n的唯一且递增的m
    strictly_increasing_mappings = {}
    last_mapped_m = -1  # 保证m是从0开始且严格递增的
    
    # 遍历 warping function
    for m, n in warping_function:
        # 只记录每个n第一次出现的m，且m必须大于之前的m值
        if n not in strictly_increasing_mappings and m > last_mapped_m:
            strictly_increasing_mappings[n] = m
            last_mapped_m = m
    
    # 将字典转换为n的排序列表，只包括唯一且递增的m
    sorted_n = sorted(strictly_increasing_mappings.keys())
    strictly_increasing_m_list = [strictly_increasing_mappings[n] for n in sorted_n]

    return strictly_increasing_m_list

def compute_cost_matrix(func1, reference_curve, t_values):
    # Compute the cost matrix between a parameterized curve and a reference curve
    # t_values is a monotonic increasing sequence in the range (0, 1)
    curve1 = func1(t_values)
    n, m = len(t_values), len(reference_curve)
    cost_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            # 计算曲线1上的点与参考曲线上的每个点之间的距离
            cost_matrix[i, j] = np.linalg.norm(curve1[i] - reference_curve[j])
    return cost_matrix

def reparameterize_curve(func, warping_path, num_points=100):
    # Reparameterize a curve using the warping path from DTW
    t_values = np.linspace(0, 1, num_points)
    curve = func(t_values)
    reparam_curve = np.array([curve[n] for n, _ in warping_path])
    return reparam_curve
