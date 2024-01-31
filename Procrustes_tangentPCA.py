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
import geomstats.geometry.pre_shape as pre_shape # 黎曼几何计算包
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
from myvtk.MyAlignment import *

# mpl.rcParams['xtick.labelsize'] = 18
# mpl.rcParams['ytick.labelsize'] = 18
# mpl.rcParams['axes.labelsize'] = 18
# sns.set_context("talk", font_scale=0.8)
# sns.set_style("whitegrid")

PCA_N_COMPONENTS = 16
Multi_plot_rows = 4
SCALETO1 = False
PCA_STANDARDIZATION = 1
ORIGINAL_GEO_PARAM = False
USE_REAL_DATA_FOR_GEO_PARAM = False

# 获取当前时间
start_time = datetime.now()
smooth_scale = 0.01
# 将时间格式化为 'yymmddhhmmss' 格式
dir_formatted_time = start_time.strftime('%y-%m-%d-%H-%M-%S')
bkup_dir = mkdir("./", "save_data_Procrustess")
bkup_dir = mkdir(bkup_dir, dir_formatted_time)
current_file_path = os.path.abspath(__file__)
current_file_name = os.path.basename(__file__)
backup_file_path = os.path.join(bkup_dir, current_file_name)
log = open(bkup_dir+"log.txt", "w")
log.write("Start at: {}\n".format(dir_formatted_time))
log.write("PCA_N_COMPONENTS:"+str(PCA_N_COMPONENTS)+"\n")
unaligned_curves = []
Files = []
radii = []
pre_Curvatures = []
pre_Torsions = []
Typevalues = [] 
# window size
# 创建离散曲线空间




pre_files = glob.glob("./scaling/resamp_attr_ascii/vmtk64a/*.vtk") # vmodeler搞完的文件
shapetype = pd.read_csv("./UVCS_class.csv", header=None)
ill=pd.read_csv("./illcases.txt",header=None) # 数据差的case，不要
ill = np.array(ill[0])
for idx in range(len(pre_files)):
    # filename = pre_files[idx].split("\\")[-1].split(".")[0][:-8]
    filename = os.path.splitext(os.path.basename(pre_files[idx]))[0][:-8]
    # print (filename)
    if filename in ill:
        print (filename, "is found in illcases.txt, skip")
        continue
    # print (filename)
    new_type_value = shapetype.loc[shapetype[0] == filename, 2].iloc[0]
    Typevalues.append(new_type_value)
    pt, Curv, Tors, Radius, Abscissas, ptns, ftangent, fnormal, fbinormal = GetMyVtk(pre_files[idx], frenet=1)
    Files.append(pre_files[idx])
    pt = pt-np.mean(pt,axis=0) # 质心归零
    # pt = interpolate_pt(pt, 640)
    # pt = pt - (pt[0]+[np.random.uniform(-0.0001,0.0001),np.random.uniform(-0.0001,0.0001),np.random.uniform(-0.0001,0.1)])
    # pt = pt/np.linalg.norm(pt[-1] - pt[0]) * 64
    unaligned_curves.append(pt[::-1]) # 反着写入，为了让看的时候是ICA->MCA
    radii.append(Radius[::-1])
    pre_Curvatures.append(Curv[::-1])
    pre_Torsions.append(Tors[::-1])
unaligned_curves = np.array(unaligned_curves)
# geometry_dir = mkdir(bkup_dir, "geometry")
Typevalues = np.array(Typevalues)

POINTS_NUM = pt.shape[0]
print ("POINTS_NUM:", POINTS_NUM)

# 初始化离散曲线空间，用一个神奇的包
r3 = Euclidean(dim=3)
srv_metric = SRVMetric(r3)
discrete_curves_space = DiscreteCurves(ambient_manifold=r3, k_sampling_points=POINTS_NUM)
print (discrete_curves_space.metric)
########################################

log.write("Data used: {}".format(len(Files)))
Files_df = pd.read_csv("./siphon_index.csv")
siphon_name = Files_df.iloc[:, 0]
pre_siphon_idx = Files_df.iloc[:, 1]
pre_siphon_arb = Files_df.iloc[:, 2]
siphon_idx = []
siphon_arb = []
for i in range(len(Files)):
    for j in range(len(siphon_name)):
        if siphon_name[j] in Files[i]:
            siphon_idx.append(63-int(pre_siphon_idx[j]))
            siphon_arb.append(pre_siphon_arb[j])
            break
    if "BG0014_R" in Files[i]:
        base_id = i
log.write("Alignment reference data (base_id):{},casename:{}".format(base_id, Files[base_id]))
# print (siphon_idx)

# 设定参考点 p，您可以根据需要修改这个点
p = np.array([0, 0, 1])  # 例如，令 p 在 z 轴上
unaligned_curves = np.array([align_endpoints(curve, p) for curve in unaligned_curves])
# align_endpoints 第一个点和最后一个点的连线和p点的连线重合

fig = plt.figure(figsize=(13, 6),dpi=300)
ax1 = fig.add_subplot(133)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(131)
for i in range(len(unaligned_curves)):
    siphon = siphon_idx[i]
    ax1.plot(unaligned_curves[i][:,0], unaligned_curves[i][:,1])
    ax2.plot(unaligned_curves[i][:,0], unaligned_curves[i][:,2])
    ax3.plot(unaligned_curves[i][:,1], unaligned_curves[i][:,2])
    # ax1.scatter(unaligned_curves[i][siphon,0], unaligned_curves[i][siphon,1])
    # ax2.scatter(unaligned_curves[i][siphon,0], unaligned_curves[i][siphon,2])
    # ax3.scatter(unaligned_curves[i][siphon,1], unaligned_curves[i][siphon,2])
fig.savefig(bkup_dir+"unaligned_curves.png")
plt.close(fig)


##################################################
#  从这里开始是对齐。                             #
#  To-Do: 需要保存Procrustes对齐后的              #
#  曲线各一条，作为后续的曲线对齐的基准。           #
##################################################

# a_curves = align_icp(unaligned_curves, base_id=base_id)
#print ("First alignment done.")
Procrustes_curves = copy.deepcopy(unaligned_curves)
temp_mean_shapes = []
interpolate_pt_num = 640
# fig = plt.figure(figsize=(13, 6),dpi=300)
# ax1 = fig.add_subplot(111)
interpolated_procrustes_curves = []
for i in range(len(Procrustes_curves)):
    interpolated_procrustes_curves.append(interpolate_pt(Procrustes_curves[i], interpolate_pt_num))
interpolated_procrustes_curves = np.array(interpolated_procrustes_curves)
for j in range(10):
    temp_mean_shape = compute_frechet_mean(interpolated_procrustes_curves)
    #  temp_mean_shape = np.array(temp_mean_shape)
    temp_mean_shape = temp_mean_shape/measure_length(temp_mean_shape)*measure_length(interpolated_procrustes_curves[0])
    print ("temp_mean_shape.shape:", temp_mean_shape.shape)
    temp_mean_shapes.append(temp_mean_shape)
    #ax1.plot(temp_mean_shape[:,0], temp_mean_shape[:,2], label="{}".format(j), marker="o")

    interpolated_procrustes_curves = align_procrustes(interpolated_procrustes_curves, base_id=-2, external_curve=temp_mean_shape)

    parametrized_curves = np.zeros_like(interpolated_procrustes_curves)
    # aligned_curves = np.zeros_like(interpolated_curves)
    for i in range(len(interpolated_procrustes_curves)):
        parametrized_curves[i] = arc_length_parametrize(interpolated_procrustes_curves[i])
    interpolated_procrustes_curves = np.array(parametrized_curves)

    ###################################
    # dynamic time warping.

    # Apply the function to the original curve using the path from dynamic programming
    reparam_curves = []
    optimal_path_dir = mkdir(bkup_dir, "optimal_path")
    for n in range(len(Procrustes_curves)):
        curve1 = interpolated_procrustes_curves[n]
        if j < 5:
            curve2 = interpolated_procrustes_curves[base_id]
        else:
            curve2 = temp_mean_shape
        # curve2 = interpolated_procrustes_curves[0]
        # original_curve = Procs_srvf_curves[n]
        # curve1 = interpolate_pt(curve1, 640)
        # curve2 = interpolate_pt(curve2, 400)
        # original_curve = interpolate_pt(original_curve, 400)
        cost_matrix = compute_cost_matrix(curve1, curve2)
        optimal_path = np.array(find_optimal_reparametrization(cost_matrix))
        # print ("optimal_path:", optimal_path)
        # ax1.plot(optimal_path[:,0], optimal_path[:,1])
        f = open(optimal_path_dir+"{:0=4}.csv".format(n), 'w', encoding='utf-8', newline='')
        # reparam_curve1 = np.zeros_like(curve1)
        # for i, j in optimal_path:
        #     # reparam_curve1[j, :] = curve1[i, :]
        #     f.write(str(i)+","+str(j)+"\n")
        #     reparam_curve1[j, :] = original_curve[i, :]
        # f.close()
        unique_mapping = extract_increasing_mappings(optimal_path)
        ax1.plot(unique_mapping)
        for i in range(len(unique_mapping)):
            f.write(str(i)+","+str(unique_mapping[i])+"\n")
        f.close()
        # if len(set(unique_mapping)) == len(unique_mapping):
        #     print ("unique_mapping has no ring.")
        #     # continue
        # else:
        #     print ("unique_mapping has rings.")
        reparam_curve1 = curve1[unique_mapping]
        if len(reparam_curve1) != interpolate_pt_num:
            reparam_curve1 = interpolate_pt(reparam_curve1, interpolate_pt_num)
        # print ("reparam_curve1.shape:", reparam_curve1.shape)
        reparam_curves.append(reparam_curve1)
    reparam_curves = np.array(reparam_curves)
    reparam_curves = align_procrustes(reparam_curves, base_id=0)
    frechet_reparam = compute_frechet_mean(reparam_curves)
    interpolated_procrustes_curves = reparam_curves


restore_pt_num = interpolate_pt_num//POINTS_NUM
Procrustes_curves = interpolated_procrustes_curves[:, ::restore_pt_num, :]
print ("Procrustes_curves.shape after alignment:", Procrustes_curves.shape)
# 可能有因素导致少一些曲线？


fig = plt.figure(figsize=(13, 6),dpi=300)
ax1 = fig.add_subplot(133)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(131)
for i in range(len(Procrustes_curves)):
    siphon = siphon_idx[i]
    ax1.plot(Procrustes_curves[i][:,0], Procrustes_curves[i][:,1])
    ax2.plot(Procrustes_curves[i][:,0], Procrustes_curves[i][:,2])
    ax3.plot(Procrustes_curves[i][:,1], Procrustes_curves[i][:,2])
fig.savefig(bkup_dir+"Procrustes_curves_final.png")
plt.close(fig)





frechet_curvature, frechet_torsion = compute_curvature_and_torsion(frechet_reparam)
frechet_energy = compute_geometry_param_energy(frechet_curvature, frechet_torsion)
frechet_tortuosity = compute_tortuosity(frechet_reparam)
print ("frechet_energy:", frechet_energy, "frechet_tortuosity:", frechet_tortuosity)
# fieldAttribute(list):
#  ['Curvature', 'float',pandas.Series],
#  ['Torsion', 'float',pandas.Series]]
scalarAttribute = [['Curvature', 'float',pd.Series(frechet_curvature)],
    ['Torsion', 'float',pd.Series(frechet_torsion)]]
# makeVtkFile(bkup_dir+"frechet_original.vtk", frechet_original, [], [])
makeVtkFile(bkup_dir+"frechet_reparam.vtk", frechet_reparam, scalarAttribute, [])

# dynamic time warping.
###################################

log.write("- preprocessing_pca is not a SRVF PCA, and it is conducted on Procrustes_cruves before aligned_endpoints.\n")
preprocessing_pca = PCAHandler(Procrustes_curves.reshape(len(Procrustes_curves),-1), None, 20, PCA_STANDARDIZATION)
preprocessing_pca.PCA_training_and_test()
preprocess_curves = preprocessing_pca.inverse_transform_from_loadings(preprocessing_pca.train_res).reshape(len(preprocessing_pca.train_res), -1, 3)

Procrustes_curves = preprocess_curves

#------------- 前处理结束 -----------------#

# log.write("Procrustes_curves is aligned again by endpoints.\n")
# Procrustes_curves = np.array([align_endpoints(curve, p) for curve in Procrustes_curves])
# SRVF计算
Procs_srvf_curves = np.zeros_like(Procrustes_curves)
np.save("Procrustes_curves.npy", Procrustes_curves)
procrustes_curves_dir = mkdir(bkup_dir, "procrustes_curves")
for i in range(len(Procrustes_curves)):
    filename = Files[i].split("\\")[-1].split(".")[0][:-8]
    makeVtkFile(procrustes_curves_dir+"{:0=4}_{}.vtk".format(i, filename), Procrustes_curves[i], [], [])


for i in range(len(Procrustes_curves)):
    # Procs_srvf_curves[i] = calculate_srvf((Procrustes_curves[i])/measure_length(Procrustes_curves[i]))
    # Procs_srvf_curves[i] = calculate_srvf((Procrustes_curves[i])/np.linalg.norm(Procrustes_curves[i][-1] - Procrustes_curves[i][0]))
    Procs_srvf_curves[i] =Procrustes_curves[i]/measure_length(Procrustes_curves[i])
    # print ("SRVF length:", measure_length(Procs_srvf_curves[i]))
    # print ("SRVF length by GPT4:", srvf_length(Procs_srvf_curves[i]))
log.write("SCALED to (absolute length) 1 and did not compute SRVF manually.\n")
log.write("according to A robust tangent PCA via shape restoration for shape variability analysis, tangent PCA is supposed to be conducted on SRVF. The convert is conducted inside Geomstats.\n")

# TANGENT_BASE = "srvf_frechet"
#TANGENT_BASE = "BH0012_R"
# if TANGENT_BASE == "frechet":
#     frechet_mean_shape = compute_frechet_mean(Procrustes_curves)
#     tangent_base = calculate_srvf(frechet_mean_shape)/measure_length(frechet_mean_shape)
#     print ("using frechet as tangent_base.")
#     log.write("using frechet as tangent_base.\n")
# elif TANGENT_BASE == "srvf_frechet":
#     tangent_base = compute_frechet_mean(Procs_srvf_curves)
#     print ("using srvf_frechet as tangent_base.")
#     log.write("using srvf_frechet as tangent_base.\n")
#     frechet_mean_shape = compute_frechet_mean(Procrustes_curves)
    # print ("tangent_base_curve length:", measure_length(tangent_base_curve))
# else:
#     for i in range(len(Files)):
#         if TANGENT_BASE in Files[i]:
#             tangent_base_id = i
#     print ("using {} as tangent_base_id.".format(Files[tangent_base_id]))
#     log.write("using {} as tangent_base_id.\n".format(Files[tangent_base_id]))
#     tangent_base = Procs_srvf_curves[tangent_base_id]


tangent_base = compute_frechet_mean(Procs_srvf_curves)
frechet_mean_shape = compute_frechet_mean(Procs_srvf_curves)


srvf_curves=[]
for i in range(len(Procs_srvf_curves)):
    srvf_curves.append(calculate_srvf(Procs_srvf_curves[i]))
frechet_mean_srvf = calculate_srvf(frechet_mean_shape)

#########################################
# 把srvf曲线做对数映射，得到切线空间的切向量
tangent_vectors = []
for curve in Procs_srvf_curves:
    tangent_vector = discrete_curves_space.to_tangent(curve, tangent_base)
    tangent_vectors.append(tangent_vector)
tangent_vectors = np.array(tangent_vectors)

# 把srvf曲线做对数映射，得到切线空间的切向量
#########################################


# srvf_pca = PCAHandler(Procs_srvf_curves.reshape(len(Procs_srvf_curves),-1), None, PCA_N_COMPONENTS, PCA_STANDARDIZATION)
# srvf_pca.PCA_training_and_test()
tpca = TangentPCA(metric=discrete_curves_space.metric, n_components=PCA_N_COMPONENTS)
# 拟合并变换数据到切线空间的主成分中
tpca.fit(tangent_vectors)
tangent_projected_data = tpca.transform(tangent_vectors)
np.save("tangent_projected_data.npy", tangent_projected_data)
with open('tpca.pkl', 'wb') as f:
    pickle.dump(tpca, f)
# reverse_tangent_vectors = tpca.inverse_transform(tangent_projected_data)

# 使用高斯方法拟合数据
tangent_pca_gaussian = fit_gaussian(tangent_projected_data)

# 从每个高斯分布中独立采样
sample_num = 500  # 定义您想要的样本数
synthetic_features = np.array([g.rvs(sample_num) for g in tangent_pca_gaussian]).T

reconstructed_curves = POINTS_NUM*from_tangentPCA_feature_to_curves(tpca, tangent_base, tangent_projected_data, PCA_N_COMPONENTS, discrete_curves_space, inverse_srvf_func=None)
reconstructed_synthetic_curves = POINTS_NUM*from_tangentPCA_feature_to_curves(tpca, tangent_base, synthetic_features, PCA_N_COMPONENTS, discrete_curves_space, inverse_srvf_func=None)


# 把主成分还原成曲线
tangent_components = reconstruct_components(tpca, discrete_curves_space, tangent_base, inverse_srvf)
for i in range(len(tangent_components)):
    tangent_components[i] = tangent_components[i]*POINTS_NUM
###########################################################

###########################################################
# Assuming tpca has an explained_variance_ratio_ attribute
explained_variance_ratio = tpca.explained_variance_ratio_
# Calculate the cumulative variance
cumulative_variance = np.cumsum(explained_variance_ratio)
# Plot the CCR curve
plt.figure(figsize=(8, 3))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
plt.title('Cumulative Captured Ratio (CCR) of PCA')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.savefig(bkup_dir+"CCR.png")
plt.close()
###########################################################

Curvatures, Torsions = compute_synthetic_curvature_and_torsion(reconstructed_curves)
synthetic_Curvatures, synthetic_Torsions = compute_synthetic_curvature_and_torsion(reconstructed_synthetic_curves)
post_reconstructed_synthetic_curves = []
post_synthetic_Curvatures = []
post_synthetic_Torsions = []
post_synthetic_features = []
count = 0
for i in range(len(reconstructed_synthetic_curves)):
    Curvature_energy, Torsion_energy = compute_geometry_param_energy(synthetic_Curvatures[i], synthetic_Torsions[i])
    Tortuosity = compute_tortuosity(reconstructed_synthetic_curves[i])
    if (0.01<Curvature_energy < 0.12) and (Torsion_energy < 0.2) and (1.6< Tortuosity < 2.7):
        post_reconstructed_synthetic_curves.append(reconstructed_synthetic_curves[i])
        post_synthetic_Curvatures.append(synthetic_Curvatures[i])
        post_synthetic_Torsions.append(synthetic_Torsions[i])
        post_synthetic_features.append(synthetic_features[i])
        count += 1
    if count == 320:
        break
    post_reconstructed_synthetic_curves.append(reconstructed_synthetic_curves[i])
    post_synthetic_Curvatures.append(synthetic_Curvatures[i])
    post_synthetic_Torsions.append(synthetic_Torsions[i])
    post_synthetic_features.append(synthetic_features[i])
    count += 1
reconstructed_synthetic_curves = np.array(post_reconstructed_synthetic_curves)
synthetic_Curvatures = np.array(post_synthetic_Curvatures)
synthetic_Torsions = np.array(post_synthetic_Torsions)
synthetic_features = np.array(post_synthetic_features)

correlation_matrix_curvature = np.corrcoef(tangent_projected_data.T, Curvatures.T)
correlation_matrix_torsion = np.corrcoef(tangent_projected_data.T, Torsions.T)
correlation_matrix_synthetic_curvature = np.corrcoef(synthetic_features.T, synthetic_Curvatures.T)
correlation_matrix_synthetic_torsion = np.corrcoef(synthetic_features.T, synthetic_Torsions.T)
print ("correlation_matrix_curvature[:16,16:].shape",correlation_matrix_curvature[:16,16:].shape)
# 绘制热图
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
sns.heatmap(correlation_matrix_curvature[:16,16:], annot=False, fmt=".2f", cmap='seismic', ax=ax1)
sns.heatmap(correlation_matrix_torsion[:16,16:], annot=False, fmt=".2f", cmap='seismic', ax=ax2)
plt.savefig(bkup_dir+"correlation_matrix.png")
plt.close(fig)

average_of_means_torsions = np.mean([np.mean(np.abs(tors)) for tors in Torsions])
average_of_std_torsions = np.mean([np.std(tors) for tors in Torsions])

torsion_param_group = []
param_group = []
HT_group = []
LT_group = []
for i in range(len(Torsions)):
    if np.mean(np.abs(Torsions[i])) > average_of_means_torsions and np.std(Torsions[i]) > average_of_std_torsions:
        torsion_param_group.append("HMHS")
        param_group.append("HT")
        HT_group.append(i)
    elif np.mean(np.abs(Torsions[i])) > average_of_means_torsions and np.std(Torsions[i]) <= average_of_std_torsions:
        torsion_param_group.append("HMLS")
        param_group.append("HT")
        HT_group.append(i)
    elif np.mean(np.abs(Torsions[i])) <= average_of_means_torsions and np.std(Torsions[i]) > average_of_std_torsions:
        torsion_param_group.append("LMHS")
        param_group.append("LT")
        LT_group.append(i)
    elif np.mean(np.abs(Torsions[i])) <= average_of_means_torsions and np.std(Torsions[i]) <= average_of_std_torsions:
        torsion_param_group.append("LMLS")
        param_group.append("LT")
        LT_group.append(i)

synthetic_torsion_param_group = []
synthetic_param_group = []
synthetic_HT_group = []
synthetic_LT_group = []
for i in range(len(synthetic_Torsions)):
    if np.mean(np.abs(synthetic_Torsions[i])) > average_of_means_torsions and np.std(synthetic_Torsions[i]) > average_of_std_torsions:
        synthetic_torsion_param_group.append("HMHS")
        synthetic_param_group.append("HT")
        synthetic_HT_group.append(i)
    elif np.mean(np.abs(synthetic_Torsions[i])) > average_of_means_torsions and np.std(synthetic_Torsions[i]) <= average_of_std_torsions:
        synthetic_torsion_param_group.append("HMLS")
        synthetic_param_group.append("HT")
        synthetic_HT_group.append(i)
    elif np.mean(np.abs(synthetic_Torsions[i])) <= average_of_means_torsions and np.std(synthetic_Torsions[i]) > average_of_std_torsions:
        synthetic_torsion_param_group.append("LMHS")
        synthetic_param_group.append("LT")
        synthetic_LT_group.append(i)
    elif np.mean(np.abs(synthetic_Torsions[i])) <= average_of_means_torsions and np.std(synthetic_Torsions[i]) <= average_of_std_torsions:
        synthetic_torsion_param_group.append("LMLS")
        synthetic_param_group.append("LT")
        synthetic_LT_group.append(i)


LT_curvatures = Curvatures[LT_group]
HT_curvatures = Curvatures[HT_group]
print ("len(LT_curvatures):", len(LT_curvatures))
print ("len(HT_curvatures):", len(HT_curvatures))
average_of_LT_curvature = np.mean([np.mean(curv) for curv in LT_curvatures])
average_of_HT_curvature = np.mean([np.mean(curv) for curv in HT_curvatures])
# average_of_LT_curvature = np.mean([np.mean(curv) for curv in Curvatures])
# average_of_HT_curvature = np.mean([np.mean(curv) for curv in Curvatures])
print ("average_of_LT_curvature:", average_of_LT_curvature)
print ("average_of_HT_curvature:", average_of_HT_curvature)
quad_param_group = []
for i in range(len(Curvatures)):
    curvature_mean = np.mean(Curvatures[i])
    if param_group[i] == 'LT':
        threshold = average_of_LT_curvature
    elif param_group[i] == 'HT':
        threshold = average_of_HT_curvature
    if curvature_mean > threshold:
        quad_param_group.append(param_group[i] + 'HC')
        param_group[i] = torsion_param_group[i] + 'HC'
    else:
        quad_param_group.append(param_group[i] + 'LC')
        param_group[i] = torsion_param_group[i] + 'LC'

synthetic_LT_curvatures = synthetic_Curvatures[synthetic_LT_group]
synthetic_HT_curvatures = synthetic_Curvatures[synthetic_HT_group]
print ("len(synthetic_LT_curvatures):", len(synthetic_LT_curvatures))
print ("len(synthetic_HT_curvatures):", len(synthetic_HT_curvatures))
synthetic_quad_param_group = []
for i in range(len(synthetic_Curvatures)):
    synthetic_curvature_mean = np.mean(synthetic_Curvatures[i])
    if synthetic_param_group[i] == 'LT':
        threshold = average_of_LT_curvature
    elif synthetic_param_group[i] == 'HT':
        threshold = average_of_HT_curvature
    if synthetic_curvature_mean > threshold:
        synthetic_quad_param_group.append(synthetic_param_group[i] + 'HC')
        synthetic_param_group[i] = synthetic_torsion_param_group[i] + 'HC'
    else:
        synthetic_quad_param_group.append(synthetic_param_group[i] + 'LC')
        synthetic_param_group[i] = synthetic_torsion_param_group[i] + 'LC'


# 给定的代码
counter = defaultdict(lambda: defaultdict(int))
overall_counter = defaultdict(int)

# 遍历每个数据点并更新计数器
for type_val, param in zip(Typevalues, param_group):
    counter[type_val][param] += 1
    overall_counter[param] += 1

# 绘制柱状图
labels = list(counter.keys())
labels.append("Overall")
type_vals = sorted(list({tv for inner_dict in counter.values() for tv in inner_dict.keys()}))

# 数据存储的列表
param2cusv_data_list = []
# 每个 Typevalue 的柱子位置
for idx, tv in enumerate(type_vals):
    counts = [counter[label][tv] for label in labels[:-1]]
    counts.append(overall_counter[tv])  # 添加全体数据的计数
    param2cusv_data_list.append(counts)

# 将数据转化为 DataFrame
param2cusv_df = pd.DataFrame(param2cusv_data_list, columns=labels, index=type_vals)
# 将数据保存为CSV文件
param2cusv_df.to_csv(bkup_dir+"param2cusv.csv")

param_group_unique_labels = list(set(quad_param_group))
# 初始化字典
param_dict = {label: {} for label in param_group_unique_labels}
synthetic_param_dict = {label: {} for label in param_group_unique_labels}
# 填充字典
for label in param_group_unique_labels:
    # 使用布尔索引来选择与当前标签对应的数据
    selected_data_torsion = [Torsions[i] for i, tag in enumerate(quad_param_group) if tag == label]
    selected_data_curvature = [Curvatures[i] for i, tag in enumerate(quad_param_group) if tag == label]
    select_data_siphon = [siphon_idx[i] for i, tag in enumerate(quad_param_group) if tag == label]
    synthetic_selected_data_torsion = [synthetic_Torsions[i] for i, tag in enumerate(synthetic_quad_param_group) if tag == label]
    synthetic_selected_data_curvature = [synthetic_Curvatures[i] for i, tag in enumerate(synthetic_quad_param_group) if tag == label]
    # 将选择的数据转换为numpy array并保存到字典中
    param_dict[label]['Torsion'] = np.array(selected_data_torsion)
    param_dict[label]['Curvature'] = np.array(selected_data_curvature)
    param_dict[label]['Siphon'] = np.array(select_data_siphon)
    synthetic_param_dict[label]['Torsion'] = np.array(synthetic_selected_data_torsion)
    synthetic_param_dict[label]['Curvature'] = np.array(synthetic_selected_data_curvature)
    # 初始化能量值列表
    energies = []
    for torsion, curvature in zip(np.array(selected_data_torsion), selected_data_curvature):
        energy = compute_geometry_param_energy(curvature, torsion)
        energies.append(energy)
    synthetic_energies = []
    for torsion, curvature in zip(np.array(synthetic_selected_data_torsion), synthetic_selected_data_curvature):
        energy = compute_geometry_param_energy(curvature, torsion)
        synthetic_energies.append(energy)
    # 将计算的能量值存储在字典中
    param_dict[label]['Energy'] = energies
    synthetic_param_dict[label]['Energy'] = synthetic_energies

    selected_data_curve = [Procrustes_curves[i] for i, tag in enumerate(quad_param_group) if tag == label]
    synthetic_selected_data_curve = [reconstructed_synthetic_curves[i] for i, tag in enumerate(synthetic_quad_param_group) if tag == label]
    tortuosity = []
    synthetic_tortuosity = []
    for curve in selected_data_curve:
        tortuosity.append(compute_tortuosity(curve))
    for curve in synthetic_selected_data_curve:
        synthetic_tortuosity.append(compute_tortuosity(curve))
    param_dict[label]['Tortuosity'] = np.array(tortuosity)
    synthetic_param_dict[label]['Tortuosity'] = np.array(synthetic_tortuosity)


param_group_unique_labels = sorted(param_group_unique_labels)
print ("param_group_unique_labels:", param_group_unique_labels)
# 定义颜色映射
colors = {
    label: plt.cm.CMRmap((i+1)/(len(param_group_unique_labels)+1)) for i, label in enumerate(param_group_unique_labels)
}

# Calculate centroids and scores for each label
energy_centroids = {}
energy_scores = {label: [] for label in param_group_unique_labels}
# Calculate the centroid of energies for each label
for label in param_group_unique_labels:
    energy_centroids[label] = compute_energy_centroid(param_dict[label]['Energy'])

# Calculate the scores for each energy value based on its distance from the centroid
fig = plt.figure(figsize=(8, 5), dpi=300)
ax = fig.add_subplot(111)
for i in range(len(Procrustes_curves)):
    energy = compute_geometry_param_energy(Curvatures[i], Torsions[i])
    ax.scatter(energy[0], energy[1], color=colors[quad_param_group[i]], alpha=0.9, s=25, marker="${}$".format(Typevalues[i]))
    # ax.annotate(Files[i].split("\\")[-1].split(".")[-2][:-7], (energy[0], energy[1]), fontsize=5)
    # print (quad_param_group[i], Typevalues[i])
    shape_score = {}
    for label in param_group_unique_labels:
        energy_score = score_energy(energy, energy_centroids[label])
        shape_score[label] = energy_score
    # print (shape_score)
    # print ("-"*20)
plt.savefig(bkup_dir+"centroid_classification.png")
plt.close()

##########################################################
# 计算距离矩阵


# 创建一个图形和轴
fig1, ax1 = plt.subplots(dpi=300)
fig2, ax2 = plt.subplots(dpi=300)
fig3 = plt.figure(figsize=(8, 5), dpi=300)
ax3 = fig3.add_subplot(111, projection='3d')
# 初始化一个索引来跟踪y_prob_max中的当前位置
index = 0
# 绘制散点图
# 定义颜色映射
Typevalues_colors = {
    label: plt.cm.gnuplot((i)/4) for i, label in enumerate(set(Typevalues))
}
param_group_colors = {
    label: plt.cm.gnuplot((i)/8) for i, label in enumerate(set(param_group))
}
total_curvature_energy = []
total_torsion_energy = []
synthetic_total_curvature_energy = []
synthetic_total_torsion_energy = []
total_tortuosity = []
synthetic_total_tortuosity = []
for label in param_group_unique_labels:
    energies = param_dict[label]['Energy']
    synthetic_energies = synthetic_param_dict[label]['Energy']
    tortuosity = param_dict[label]['Tortuosity']
    total_tortuosity.extend(tortuosity)
    synthetic_tortuosity = synthetic_param_dict[label]['Tortuosity']
    synthetic_total_tortuosity.extend(synthetic_tortuosity)
    curvatures, torsions = zip(*energies)
    synthetic_curvatures, synthetic_torsions = zip(*synthetic_energies)
    total_curvature_energy.extend(curvatures)
    total_torsion_energy.extend(torsions)
    synthetic_total_curvature_energy.extend(synthetic_curvatures)
    synthetic_total_torsion_energy.extend(synthetic_torsions)
    # 获取当前标签对应的大小值
    # sizes_for_label = y_prob_max[index : index + len(energies)]
    ax1.scatter(curvatures, torsions, 
               color=colors[label], 
               label=label, 
               alpha=0.9, 
               s = tortuosity*tortuosity*20) 
    fontsize = 5
    for i in range(len(curvatures)):
        ax1.annotate(round(tortuosity[i],2), (curvatures[i], torsions[i]), fontsize=fontsize)
    ax2.scatter(synthetic_curvatures, synthetic_torsions,
                color=colors[label],
                label=label,
                alpha=0.5,
                s = synthetic_tortuosity*synthetic_tortuosity*20)
    # for i in range(len(curvatures)):
    #     fontsize = 5
    #     ax1.annotate(Files[i].split("\\")[-1].split(".")[-2][:-7], (curvatures[i], torsions[i]), fontsize=fontsize)
    #     ax1.annotate(param_group[i], (curvatures[i], torsions[i]-0.0015), fontsize=fontsize, color=param_group_colors[param_group[i]])
    #     ax1.annotate(Typevalues[i], (curvatures[i], torsions[i]-0.0030), fontsize=fontsize, color=Typevalues_colors[Typevalues[i]] )
    # # 更新索引
    ax3.scatter3D(synthetic_curvatures,synthetic_torsions,synthetic_tortuosity,
                color=colors[label],
                label=label,
                alpha=0.3,
                marker='^')
    ax3.scatter3D(curvatures,torsions,tortuosity,
                color=colors[label],
                label=label,
                alpha=0.9,
                marker='o')
    index += len(energies)

total_curvature_energy = np.array(total_curvature_energy)
total_torsion_energy = np.array(total_torsion_energy)
synthetic_total_curvature_energy = np.array(synthetic_total_curvature_energy)
synthetic_total_torsion_energy = np.array(synthetic_total_torsion_energy)
total_tortuosity = np.array(total_tortuosity)
synthetic_total_tortuosity = np.array(synthetic_total_tortuosity)
# ax2.scatter(total_curvature_energy, total_torsion_energy, color="k", alpha=0.5, s=25)
# 计算线性回归
slope, intercept, r_value, p_value, std_err = stats.linregress(total_curvature_energy, total_torsion_energy)
synthetic_slope, synthetic_intercept, synthetic_r_value, synthetic_p_value, synthetic_std_err = stats.linregress(synthetic_total_curvature_energy, synthetic_total_torsion_energy)
# 计算预测值和置信区间
x_pred = np.linspace(total_curvature_energy.min(), total_curvature_energy.max(), 100)
synthetic_x_pred = np.linspace(synthetic_total_curvature_energy.min(), synthetic_total_curvature_energy.max(), 100)
y_pred = intercept + slope * x_pred
synthetic_y_pred = synthetic_intercept + synthetic_slope * synthetic_x_pred
y_err = std_err * np.sqrt(1/len(total_curvature_energy) + (x_pred - np.mean(total_curvature_energy))**2 / np.sum((total_curvature_energy - np.mean(total_curvature_energy))**2))
conf_interval_upper = y_pred + 1.96 * y_err  # 95% 置信区间
conf_interval_lower = y_pred - 1.96 * y_err  # 95% 置信区间
ax1.plot(x_pred, y_pred, color="k", alpha=0.4, label='Fit: y = {:.2f} + {:.2f}x'.format(intercept, slope))
ax1.fill_between(x_pred, conf_interval_lower, conf_interval_upper, color='silver', alpha=0.15)# , label='95% CI')
# 比较合成形状
ax2.plot(x_pred, y_pred, color="k", alpha=0.4, linestyle=":",label='Fit: y = {:.2f} + {:.2f}x'.format(intercept, slope))
ax2.plot(synthetic_x_pred, synthetic_y_pred, color="k", alpha=0.4,label='Fit: y = {:.2f} + {:.2f}x'.format(synthetic_intercept, synthetic_slope))
font_size = 14
for ax in [ax1,ax2]:
    ax.set_xlabel('Bending Energy', fontsize=font_size)
    ax.set_ylabel('Twisting Energy', fontsize=font_size)
    # ax.set_title('Energy Scatter Plot by Label')
    # ax.set_ylim(0, 0.2)
    ax.grid(linestyle='dotted', alpha=0.5)
ax1.legend()
ax2.legend()
ax3.legend()
ax3.set_xlabel('Curvature')
ax3.set_ylabel('Torsion')
ax3.set_zlabel('Tortuosity')
ax1.tick_params(axis='both', labelsize=font_size)
ax2.tick_params(axis='both', labelsize=font_size)
ax3.tick_params(axis='both', labelsize=font_size)
# fig3.show()
fig1.savefig(bkup_dir+"Energy_Scatter_Plot_by_Label.png")
fig2.savefig(bkup_dir+"Energy_Scatter_Plot_by_Label=synthetic.png")
fig3.savefig(bkup_dir+"Tortuosity_Scatter_Plot_by_Label.png")
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)

df = pd.DataFrame(tangent_projected_data, columns=[f'PC{i+1}' for i in range(PCA_N_COMPONENTS)])
df['Type'] = quad_param_group
# 创建一个4x4的子图网格
fig, axes = plt.subplots((PCA_N_COMPONENTS//Multi_plot_rows), Multi_plot_rows, figsize=(16, 20))
# 为每个主成分绘制violinplot
for i in range(PCA_N_COMPONENTS):
    ax = axes[i // Multi_plot_rows, i % Multi_plot_rows]
    # sns.violinplot(x='Type', y=f'PC{i+1}', data=df, ax=ax, inner='quartile', palette=colors)  # inner='quartile' 在violin内部显示四分位数
    sns.boxplot(x='Type', y=f'PC{i+1}', data=df, ax=ax, palette=colors, width=0.25)  # inner='quartile' 在violin内部显示四分位数
    ax.set_title(f'MoD {i+1}')
    ax.set_ylabel('')  # 移除y轴标签，使得图更加简洁
    ax.tick_params(axis='both', labelsize=14, rotation=45)
fig.tight_layout()
plt.savefig(bkup_dir+"tangentPCA_total_boxplot.png")
plt.close()


####################BraVa Data的各PC####################
def map_quad_params_to_int(quad_params):
    unique_params, inverse_indices = np.unique(quad_params, return_inverse=True)
    # 打印每个 unique param 对应的 quad_param_group 中的元素
    mapping = {unique_param: quad_param for unique_param, quad_param in zip(unique_params, range(len(unique_params)))}
    return inverse_indices, mapping
# 假设 quad_param_group 是一个已经定义的数组
quad_param_group_mapped, mapping = map_quad_params_to_int(quad_param_group)
# 打印映射结果
print("Mapped Indices(quad_param_group_mapped): ", quad_param_group_mapped)
print("Mapping(mapping): ", mapping)
vtk_dir = mkdir(bkup_dir , "vtk")



PC_sigma_dir = mkdir(bkup_dir , "PC_sigma")
PC_gaussian_dir = mkdir(bkup_dir , "PC_gaussian")
PC_geometry_dir = mkdir(bkup_dir , "PC_geometry")
fig5 = plt.figure(figsize=(7, 6), dpi=300)
ax51 = fig5.add_subplot(131)
ax52 = fig5.add_subplot(132)
ax53 = fig5.add_subplot(133)
# ax51.plot(frechet_mean_shape[:,0],color="k")
# ax52.plot(frechet_mean_shape[:,1],color="k")
# ax53.plot(frechet_mean_shape[:,2],color="k")
MoD_cmap = plt.cm.get_cmap('jet', PCA_N_COMPONENTS)
for i in range(PCA_N_COMPONENTS):
    single_component_feature = np.zeros_like(tangent_projected_data)
    single_component_feature[:,i] = tangent_projected_data[:,i]
    single_reconstructed_curves = []
    sigma = np.std(single_component_feature[:,i])
    sigma_single_feature = np.zeros_like(tangent_projected_data)[:7]
    for j in range(7):
        sigma_single_feature[j,i] = (j-3) * sigma
    sigma_single_reconstructed_curves = POINTS_NUM*from_tangentPCA_feature_to_curves(tpca, tangent_base, sigma_single_feature, PCA_N_COMPONENTS, discrete_curves_space, inverse_srvf_func=None)
    sigma_single_reconstructed_curves = np.array([align_endpoints(sigma_single_reconstructed_curves[j], p) for j in range(7)])
    fig = plt.figure(figsize=(10,2),dpi=300)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig3 = plt.figure(figsize=(8, 5), dpi=300)
    ax31 = fig3.add_subplot(111)
    fig4 = plt.figure(figsize=(8, 5), dpi=300)
    ax41 = fig4.add_subplot(311)
    ax42 = fig4.add_subplot(312)
    ax43 = fig4.add_subplot(313)
    single_reconstructed_curves = POINTS_NUM*from_tangentPCA_feature_to_curves(tpca, tangent_base, single_component_feature, PCA_N_COMPONENTS, discrete_curves_space, inverse_srvf_func=None)
    single_reconstructed_curvature, single_reconstructed_torsion = compute_synthetic_curvature_and_torsion(single_reconstructed_curves)
    # print ("single_reconstructed_curvature.shape:", single_reconstructed_curvature.shape)
    # print ("Curvautres.shape:", Curvatures.shape)
    curvature_mean_values = np.mean(single_reconstructed_curvature, axis=0)
    curvature_std_dev = np.std(single_reconstructed_curvature, axis=0)
    torsion_mean_values = np.mean(single_reconstructed_torsion, axis=0)
    torsion_std_dev = np.std(single_reconstructed_torsion, axis=0)
    ax1.plot(curvature_mean_values, label=r'$\bar{\kappa}$', marker='o')
    ax2.plot(torsion_mean_values, label=r'$\bar{\tau}$', marker='o')
    ax1.errorbar(range(len(curvature_mean_values)), curvature_mean_values, yerr=curvature_std_dev, fmt='-o')
    ax2.errorbar(range(len(torsion_mean_values)), torsion_mean_values, yerr=torsion_std_dev, fmt='-o')
    all_single_curvatures = [curvature for curve in single_reconstructed_curvature for curvature in curve]
    all_single_torsions = [torsion for curve in single_reconstructed_torsion for torsion in curve]
    all_curvature = [curvature for curve in Curvatures for curvature in curve]
    all_torsion = [torsion for curve in Torsions for torsion in curve]
    vtk_file_name = vtk_dir + f"PC{i+1}.vtk"
    write_vtk_line(vtk_file_name, 
                   single_reconstructed_curves, 
                   quad_param_group_mapped, 
                   single_component_feature[:,i], 
                   all_single_curvatures, all_single_torsions, all_curvature, all_torsion)
    for j in range(7):
        ax41.plot(sigma_single_reconstructed_curves[j][:,0], label="{}SIGMA".format(j-3), marker='o')
        ax42.plot(sigma_single_reconstructed_curves[j][:,1], label="{}SIGMA".format(j-3), marker='o')
        ax43.plot(sigma_single_reconstructed_curves[j][:,2], label="{}SIGMA".format(j-3), marker='o')
        ax51.plot(sigma_single_reconstructed_curves[j][:,0], range(len(sigma_single_reconstructed_curves[j])), 
                color=MoD_cmap(i/PCA_N_COMPONENTS),
                alpha=0.8)
        ax52.plot(sigma_single_reconstructed_curves[j][:,1], range(len(sigma_single_reconstructed_curves[j])),
                color=MoD_cmap(i/PCA_N_COMPONENTS), 
                alpha=0.8)
        ax53.plot(sigma_single_reconstructed_curves[j][:,2], range(len(sigma_single_reconstructed_curves[j])),
                  color=MoD_cmap(i/PCA_N_COMPONENTS), 
                  alpha=0.8)
    ax51.fill_betweenx(range(len(sigma_single_reconstructed_curves[0])), 
                       sigma_single_reconstructed_curves[0][:,0], sigma_single_reconstructed_curves[-1][:,0], color=MoD_cmap(i/PCA_N_COMPONENTS), alpha=0.2)
    ax52.fill_betweenx(range(len(sigma_single_reconstructed_curves[0])),
                          sigma_single_reconstructed_curves[0][:,1], sigma_single_reconstructed_curves[-1][:,1], color=MoD_cmap(i/PCA_N_COMPONENTS), alpha=0.2)
    ax53.fill_betweenx(range(len(sigma_single_reconstructed_curves[0])),
                            sigma_single_reconstructed_curves[0][:,2], sigma_single_reconstructed_curves[-1][:,2], color=MoD_cmap(i/PCA_N_COMPONENTS), alpha=0.2)
    
    for label,key in zip(mapping.values(), mapping.keys()):
    # 筛选出特定标签的数据
        data = single_component_feature[:, i][np.array(quad_param_group_mapped) == label]
        mean_label = np.mean(data)
        std_label = np.std(data)
        # 创建用于绘制高斯分布的数据点
        x_values = np.linspace(min(data), max(data), 1000)
        y_values = norm.pdf(x_values, loc=mean_label, scale=std_label)
        # 绘制高斯分布
        ax31.plot(x_values, y_values, label=f'{key} Gaussian', color=colors[key])

    ax31.legend()
    ax43.legend()
    ax1.set_title(f'PC{i+1}')
    ax2.set_title(f'PC{i+1}')
    ax1.set_xlabel('Sampling Points')
    ax2.set_xlabel('Sampling Points')
    ax1.set_ylabel('Curvature')
    ax2.set_ylabel('Torsion')
    fig.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig.savefig(PC_geometry_dir+f"PC{i+1}.png")
    fig3.savefig(PC_gaussian_dir+f"PC{i+1}_gaussian.png")
    fig4.savefig(PC_sigma_dir+f"PC{i+1}_sigma.png")
    plt.close(fig3)
    plt.close(fig)
    plt.close(fig4)
ax52.set_yticklabels([])
ax53.set_yticklabels([])
ax51.grid(linestyle='dotted', alpha=0.9)
ax52.grid(linestyle='dotted', alpha=0.9)
ax53.grid(linestyle='dotted', alpha=0.9)
fig5.tight_layout()
fig5.savefig(PC_sigma_dir+f"PC_sigma_all.png")
plt.close(fig5)






for i in range(PCA_N_COMPONENTS):
    single_component_feature = np.zeros_like(tangent_projected_data)
    single_component_feature[:,i] = tangent_projected_data[:,i]
    single_reconstructed_curves = []
    sigma = np.std(single_component_feature[:,i])
    sigma_single_feature = np.zeros_like(tangent_projected_data)[:7]
    for j in range(7):
        sigma_single_feature[j,i] = (j-3) * sigma
    sigma_single_reconstructed_curves = POINTS_NUM*from_tangentPCA_feature_to_curves(tpca, tangent_base, sigma_single_feature, PCA_N_COMPONENTS, discrete_curves_space, inverse_srvf_func=None)
    sigma_single_reconstructed_curves = np.array([align_endpoints(sigma_single_reconstructed_curves[j], p) for j in range(7)])
    block_points = vtk.vtkPoints()
    for curve in sigma_single_reconstructed_curves:
        for p in curve:
            block_points.InsertNextPoint(p.tolist())  # 假设每个 point 是一个长度为3的列表或数组

    # 创建 vtkCellArray 对象并添加线
    block_lines = vtk.vtkCellArray()
    p_id = 0
    for curve in sigma_single_reconstructed_curves:
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(curve))
        for j in range(len(curve)):
            polyline.GetPointIds().SetId(j, p_id)
            p_id += 1
        block_lines.InsertNextCell(polyline)
    # 创建 vtkPolyData 对象
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(block_points)
    polyData.SetLines(block_lines)

    # 创建 vtkDelaunay3D 过滤器来三角化点集
    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(polyData)
    delaunay.Update()  # 确保调用 Update 方法

    # 创建文件写入器
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(vtk_dir+"curves_space_{}.vtk".format(i+1))
    writer.SetInputData(delaunay.GetOutput())

    # 写入文件
    writer.Write()




################################
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(10, 10), dpi=300)
# 设置GridSpec布局
gs = GridSpec(4, 4, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])
ax7 = fig.add_subplot(gs[3, 0])
ax8 = fig.add_subplot(gs[3, 1])
ax9 = fig.add_subplot(gs[:, 2:])
axes_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
def find_first_occurrences(mapping, labels):
    first_occurrences = {label: None for label in labels}
    for index, value in enumerate(mapping):
        if value in labels and first_occurrences[value] is None:
            first_occurrences[value] = index
        # 当所有标签都找到时，提前退出循环
        if all(first_occurrences[label] is not None for label in labels):
            break
    return [first_occurrences[label] for label in labels]

example_idx = find_first_occurrences(quad_param_group_mapped, [0,1,2,3])
print ("example_idx:", example_idx)
# example_idx = [2,6, 3, 0]
example_feature = tangent_projected_data[example_idx]
example_reconstructed_curves = []
example_reconstructed_curves = POINTS_NUM*from_tangentPCA_feature_to_curves(tpca, tangent_base, 
                                                                            example_feature, 
                                                                            PCA_N_COMPONENTS, 
                                                                            discrete_curves_space, 
                                                                            inverse_srvf_func=None)
example_reconstructed_curves = np.array([align_endpoints(example_reconstructed_curves[j], p) for j in range(len(example_reconstructed_curves))])
single_reconstructed_curvature, single_reconstructed_torsion = compute_synthetic_curvature_and_torsion(single_reconstructed_curves)
print ("example:", np.array(quad_param_group)[example_idx])
print ("example:", np.array(quad_param_group_mapped)[example_idx])
markers = ['o', '^', 's', 'd']
for i in range(len(example_idx)):
    ax9.plot(example_reconstructed_curves[i][:,0],example_reconstructed_curves[i][:,2], 
             label="{}-{}".format(i+1, quad_param_group[example_idx[i]]),
             marker=markers[i], 
             color=colors[quad_param_group[example_idx[i]]])
ax9.legend()
for i, ax in zip(range(8), axes_list):
    print ("MoD:", i+1)
    ax.set_title("MoD {}".format(i+1))

    for j in range(len(example_idx)):
        ax.axvline(x=example_feature[j][i], color=colors[quad_param_group[example_idx[j]]],
                   linestyle='--', marker=markers[j])

    for label,key in zip(mapping.values(), mapping.keys()):
    # 筛选出特定标签的数据
        # data = single_component_feature[:, i][np.array(quad_param_group_mapped) == label]
        data = tangent_projected_data[np.array(quad_param_group_mapped) == label,i]
        mean_label = np.mean(data)
        std_label = np.std(data)
        print ("label:", label, "key:", key, "mean_label:", mean_label, "std_label:", std_label)
        # 创建用于绘制高斯分布的数据点
        x_values = np.linspace(min(data), max(data), 1000)
        y_values = norm.pdf(x_values, loc=mean_label, scale=std_label)
        # 绘制高斯分布
        ax.plot(x_values, y_values, label=f'{key} Gaussian', color=colors[key])
        ax.tick_params(axis='both', labelsize=12)
        # ax_id = ax_id + 1
# ax.legend()
fig.tight_layout()
fig.savefig(PC_gaussian_dir+f"example_PC_gaussians.png")
fig.tight_layout()
plt.close(fig)



################################
    

def darken_color(color, factor=0.7):
    """使颜色变深"""
    return tuple([factor * c for c in color[:3]] + [color[3]])

reconstructed_curves = np.array([align_endpoints(curve, p) for curve in reconstructed_curves])
fig1 = plt.figure(figsize=(6,12),dpi=300)
ax1 = fig1.add_subplot(211)
ax2 = fig2.add_subplot(212)
# frechet_mean_shape = align_endpoints(frechet_mean_shape, p)
# ax.plot(frechet_mean_shape[:,0], label="Mean shape", color="k")  
curvature_means = {key: [] for key in mapping.keys()}
torsion_means = {key: [] for key in mapping.keys()}
for label,key in zip(mapping.values(), mapping.keys()):
    data = reconstructed_curves[np.array(quad_param_group_mapped) == label]
    curvature = Curvatures[np.array(quad_param_group_mapped) == label]
    torsion = Torsions[np.array(quad_param_group_mapped) == label]
    tortuosity = param_dict[key]['Tortuosity']
    C_energy = param_dict[key]['Energy'][0]
    T_energy = param_dict[key]['Energy'][1]
    print ("label:", label, "key:", key)
    print ("Max Curvature:", np.max(curvature.flatten()))
    #print ("Min Curvature:", np.min(curvature.flatten()))
    print ("Avg Curvature:", np.mean(curvature.flatten()))
    print ("Max Torsion:", np.max(torsion.flatten()))
    # print ("Min Torsion:", np.min(torsion.flatten()))
    print ("Avg Torsion:", np.mean(torsion.flatten()))
    print ("Max Tortuosity:", np.max(tortuosity))
    print ("Min Tortuosity:", np.min(tortuosity))
    print ("Avg Tortuosity:", np.mean(tortuosity))
    print ("Max curvature Energy:", np.max(C_energy))
    print ("Min curvature Energy:", np.min(C_energy))
    print ("Avg curvature Energy:", np.mean(C_energy))
    print ("Max torsion Energy:", np.max(T_energy))
    print ("Min torsion Energy:", np.min(T_energy))
    print ("Avg torsion Energy:", np.mean(T_energy))
    print ("-"*20)

    # for curve in data:
    #     ax.scatter(range(len(curve)), curve[:,0], color=colors[key], alpha=0.5, marker="+", s=10)
    # print ("colorkey", colors[key])
    # 初始化存储均值的列表
    curvature_means_per_loci = []
    torsion_means_per_loci = []
    for loci in range(len(curvature[0])):
        curvature_means[key].append(np.mean(curvature[:, loci]))
        torsion_means[key].append(np.mean(torsion[:, loci]))
        boxprops = dict(facecolor=colors[key], color=colors[key])
        medianprops = dict(color=darken_color(colors[key]), linewidth=2)
        box = ax1.boxplot(curvature[:, loci], positions=[loci], widths=0.3, 
                        showfliers=False, patch_artist=True,
                        boxprops=boxprops, medianprops=medianprops)
        box = ax2.boxplot(torsion[:, loci], positions=[loci], widths=0.3, 
                        showfliers=False, patch_artist=True,
                        boxprops=boxprops, medianprops=medianprops)
ax1.set_xlabel('Sampling Points')
ax1.set_ylabel('Curvature')
ax1.set_xticks(range(0, len(curvature[0]), 10))
ax1.set_xticklabels(ax1.get_xticks(), rotation=45)
ax2.set_xticks(range(0, len(torsion[0]), 10))
ax2.set_xlabel('Sampling Points')
ax2.set_ylabel('Torsion')
ax2.set_xticklabels(ax2.get_xticks(), rotation=45)
fig1.tight_layout()
fig1.savefig(bkup_dir+"X_axis_of_curvature_and_torsion.png")
plt.close(fig1)


# 初始化存储均值的列表
curvature_means_per_loci = [[] for _ in range(len(curvature[0]))]
torsion_means_per_loci = [[] for _ in range(len(torsion[0]))]

# 聚合不同类别的均值
for loci in range(len(curvature[0])):
    for key in curvature_means:
        curvature_means_per_loci[loci].append(curvature_means[key][loci])
        torsion_means_per_loci[loci].append(torsion_means[key][loci])

# 计算类别均值的标准差
curvature_std_devs = [np.std(means) for means in curvature_means_per_loci]
torsion_std_devs = [np.std(means) for means in torsion_means_per_loci]

# 标准差降序排序并获取前10个最大的loci
# top10_diff_curvature_loci = np.argsort(curvature_std_devs)[::-1][:10]
# top10_diff_torsion_loci = np.argsort(torsion_std_devs)[::-1][:10]

diff_curvature_loci = np.argsort(curvature_std_devs)
diff_torsion_loci = np.argsort(torsion_std_devs)


vtk_points = np.zeros((len(reconstructed_curves)+1, len(reconstructed_curves[0]), 3))
print ("vtk_points.shape:", vtk_points.shape)
vtk_groups = np.zeros((len(reconstructed_curves)+1, len(reconstructed_curves[0])))
vtk_groups[:] = -1
vtk_variance_curvature = np.zeros((len(reconstructed_curves)+1, len(reconstructed_curves[0])))
vtk_variance_torsion = np.zeros((len(reconstructed_curves)+1, len(reconstructed_curves[0])))
i = 0
for label,key in zip(mapping.values(), mapping.keys()):
    data = reconstructed_curves[np.array(quad_param_group_mapped) == label]
    for curve in data:
        for loci in range(len(curve)):
            vtk_points[i+1, loci] = curve[loci]
            vtk_groups[i+1, loci] = label
            vtk_variance_curvature[i+1][loci] = diff_curvature_loci[loci]
            vtk_variance_torsion[i+1][loci] = diff_torsion_loci[loci]
        i += 1

file_name = bkup_dir + "output.vtk"
save_vtk_file(vtk_points, vtk_groups, vtk_variance_curvature, vtk_variance_torsion, file_name)

# for loci in range(2, len(curve)-5):
#     ax1.scatter(frechet_mean_shape[loci,0], frechet_mean_shape[loci,2], marker='o',color="dimgray",
#                 alpha=np.exp(-15 * np.where(diff_curvature_loci == loci)[0]/len(diff_curvature_loci)))
#     ax2.scatter(frechet_mean_shape[loci,0], frechet_mean_shape[loci,2], marker='s',color="dimgray",
#                 alpha=np.exp(-15 * np.where(diff_torsion_loci == loci)[0]/len(diff_torsion_loci)))

fig = plt.figure(figsize=(8, 4),dpi=300)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
frechet_curvature, frechet_torsion = compute_curvature_and_torsion(frechet_reparam)
for i in range(len(reconstructed_curves[0])):
    # curvature_loci = Curvatures[:,i]-np.mean(Curvatures[:,i])
    # torsion_loci = Torsions[:,i]-np.mean(Torsions[:,i])
    curvature_loci = Curvatures[:,i]-frechet_curvature[i]
    torsion_loci = Torsions[:,i]-frechet_torsion[i]
    ax1.boxplot(curvature_loci, positions=[i], widths=0.3, showfliers=False)
    ax2.boxplot(torsion_loci, positions=[i], widths=0.3, showfliers=False)
# 假设reconstructed_curves[0]的长度是n
n = len(reconstructed_curves[0])
tick_positions = list(range(0, n+1, 8))  # 从0开始，到n结束，步长为10
scaled_tick_positions = [round(pos / n, 2) for pos in tick_positions]  # 将刻度位置从0到64的范围映射到0到1.0

for ax in [ax1, ax2]:
    ax.set_xticks(tick_positions)  # 设置原始的刻度位置
    ax.set_xticklabels(scaled_tick_positions, rotation=45)  # 显示映射后的刻度标签
    ax.grid(linestyle='dotted', alpha=0.3)


fig.tight_layout()
plt.savefig(bkup_dir+"influence_on_geometry.png")
plt.close(fig)


fig = plt.figure(figsize=(8, 4),dpi=300)
ax = fig.add_subplot(111)
for i in range(PCA_N_COMPONENTS):
    ax.boxplot(tangent_projected_data[:,i], positions=[i], widths=0.6, showfliers=False)
ax.set_xticks(range(PCA_N_COMPONENTS))
ax.set_xticklabels([f'PC{i+1}' for i in range(PCA_N_COMPONENTS)],  rotation=45)
ax.set_xlabel('Principal Components')
ax.set_ylabel('Loadings')
ax.set_title('Loadings of Principal Components')
fig.tight_layout()
plt.savefig(bkup_dir+"Loadings_of_Principal_Components.png")
plt.close(fig)

from sklearn.linear_model import RANSACRegressor
fig = plt.figure(dpi=300,figsize=(8, 4))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
x_pc = 0
y_pc = 1
s_pc = 2
for label in param_group_unique_labels:
    x_data = tangent_projected_data[np.array(quad_param_group) == label][:,x_pc]
    y_data = tangent_projected_data[np.array(quad_param_group) == label][:,y_pc]
    s_data = tangent_projected_data[np.array(quad_param_group) == label][:,s_pc]-np.min(tangent_projected_data[np.array(quad_param_group) == label][:,s_pc])+0.1
    ransac_xy = RANSACRegressor()
    ransac_xy.fit(x_data.reshape(-1, 1), y_data)
    line_x = np.arange(x_data.min(), x_data.max()).reshape(-1, 1)
    line_y = ransac_xy.predict(line_x)
    ransac_xz = RANSACRegressor()
    ransac_xz.fit(x_data.reshape(-1, 1), s_data)
    line_s = ransac_xz.predict(line_x)
    ax.scatter(x_data, y_data, 
               color=colors[label], 
               label=label, 
               alpha=0.7,)
    ax2.scatter(x_data, s_data,
                color=colors[label],
                label=label,
                alpha=0.7,)
    ax.plot(line_x, line_y, color=colors[label], alpha=0.3)
    ax2.plot(line_x, line_s, color=colors[label], alpha=0.3)
    for i in range(10):
        # makeVtkFile(vtk_dir + label + "_synthetic_" + str(i) + ".vtk",reconstructed_synthetic_curves[np.array(synthetic_quad_param_group) == label][i], [],[])
        makeVtkFile(vtk_dir + label + "_real_" + str(i) + ".vtk",reconstructed_curves[np.array(quad_param_group) == label][i],[],[])
ax.set_xlabel('MoD{}'.format(x_pc+1))
ax.set_ylabel('MoD{}'.format(y_pc+1))
ax2.set_xlabel('MoD{}'.format(x_pc+1))
ax2.set_ylabel('MoD{}'.format(s_pc+1))
ax2.legend()
ax.grid(linestyle='dotted', alpha=0.5)
ax2.grid(linestyle='dotted', alpha=0.5)
fig.tight_layout()
fig.savefig(bkup_dir + "MoDloadings_plot_both.png")
plt.close(fig1)




# 源文件和文件夹的路径
source_code = 'D:/!BraVa_src/src/ModeDecomposition/Procrustes_tangentPCA.py'
source_folder = 'D:/!BraVa_src/src/ModeDecomposition/myvtk'
# 目标路径
destination_code = os.path.join(bkup_dir, 'code.py')
destination_folder = os.path.join(bkup_dir, 'myvtk')
# 复制文件
shutil.copy(source_code, destination_code)
# 复制整个文件夹（包括内容）
shutil.copytree(source_folder, destination_folder)

end_time = datetime.now()
total_time = end_time - start_time
print(dir_formatted_time, "is done in", total_time.seconds, "seconds.")
log.close()
open_folder_in_explorer(bkup_dir)
