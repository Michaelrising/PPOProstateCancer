import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _utils import *
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
import plotly.graph_objects as go

clustering = pd.read_csv('./kmeans_clustering_results.csv', header=0, names=['y', 'r_HD', 'r_HI', 'Gamma', 'cluster'])
group1, group2 = [np.array(clustering.loc[clustering['cluster'] == i].index) for i in [0,1]]
# group2_1 = np.array(clustering.loc[clustering['cluster'] == 1].loc[clustering['x_r'] > 0].index)
# group2_2 = np.array(clustering.loc[clustering['cluster'] == 1].loc[clustering['x_r'] <= 0].index)

comparision_list = [85, 13, 88, 15 ,66, 61]

states_file_list = ['../PPO_states/analysis/patient0' + str(i)+ "_converge_high_reward_states.csv" for i in comparision_list]
c_HI_list = []
for i, file in enumerate(states_file_list):
    patient = comparision_list[i]
    list_df = pd.read_csv('../GLV/analysis-sigmoid/model_pars/patient0'+ str(patient)+'/Args_1-patient0' + str(patient) + '.csv')
    K = np.array(list_df.loc[1, ~np.isnan(list_df.loc[1, :])])
    states = pd.read_csv(file, header=0, names=['HD', 'HI', 'PSA'])
    c_HI_list.append(np.array(states['HI'])/K[1])

cs = sns.color_palette('Paired')
plt.style.use(['science', 'nature'])
colors = [cs[1], cs[1], cs[1], cs[3], cs[3], cs[3]]
lines = ['-','--','-.', '-', '--', '-.']
fig, ax = plt.subplots(figsize=[10, 8])
for i in range(len(comparision_list)):
    x = np.arange(c_HI_list[i].shape[0]) * 28
    ax.plot(x, c_HI_list[i], label='patient0' + str(comparision_list[i]), c=colors[i], ls=lines[i])
    ax.set_xlim(-100, 4500)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Time (Days)', fontsize = 22)
plt.ylabel("Cell concentration", fontsize = 22)
plt.legend(fontsize = 16)

# plt.style.use('default')
plt.style.use(['science', 'nature'])
axins = ax.inset_axes([2300, 0.05, 2000, 0.3], transform=ax.transData)
for i in range(3, 6):
    x = np.arange(c_HI_list[i].shape[0]) * 28
    axins.plot(x, c_HI_list[i], label='patient0' + str(comparision_list[i]), c=colors[i], ls=lines[i])
    axins.text(x = -100, y = 20 * 1e-4, s='$10^{-4}$', fontsize = 9)
    axins.set_xticklabels([])
    axins.set_yticks(ticks=[0, 3 * 1e-4, 6* 1e-4, 9* 1e-4, 12* 1e-4, 15 * 1e-4, 18*1e-4 ])#, labels=[0, 3, 6, 9,12, 15, 18], fontsize = 12)
plt.savefig('./6_Chosen_clustered_patients_HI_analysis.png', dpi=300)
plt.show()

## Resistance re-clustering ## group2
group2 = clustering.loc[clustering['cluster'] == 1]
normalized_group2, norm = normalize(group2[['r_HD', 'r_HI', 'Gamma']], axis = 0,return_norm=True, norm="l2")
normalized_group2_for_kmeans = pd.DataFrame({'y': list(group2.y),
                                           'r_HD': normalized_group2[:, 0],
                                           'r_HI': normalized_group2[:, 1],
                                           'Gamma': normalized_group2[:, 2]}, index=group2.index)

k_num = 2
kmeans = KMeans(n_clusters=k_num).fit(normalized_group2_for_kmeans[['r_HD', 'r_HI', 'Gamma']])
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids]
cen_y = [i[1] for i in centroids]
cen_z = [i[2] for i in centroids]
normalized_group2_for_kmeans['cluster'] = kmeans.predict(normalized_group2_for_kmeans[['r_HD', 'r_HI', 'Gamma']])
normalized_group2_for_kmeans['cen_x'] = normalized_group2_for_kmeans.cluster.map({i: cen_x[i] for i in range(k_num)})
normalized_group2_for_kmeans['cen_y'] = normalized_group2_for_kmeans.cluster.map({i: cen_y[i] for i in range(k_num)})
normalized_group2_for_kmeans['cen_z'] = normalized_group2_for_kmeans.cluster.map({i: cen_z[i] for i in range(k_num)})

colors = [cs[2*i+1] for i in range(k_num)]
symbols = ['circle' for _ in range(k_num)]
normalized_group2_for_kmeans['c'] = normalized_group2_for_kmeans.cluster.map({i: colors[i] for i in range(k_num)})
normalized_group2_for_kmeans['symbol'] = normalized_group2_for_kmeans.cluster.map({i: symbols[i] for i in range(k_num)})
original_cents = centroids * norm
from scipy.spatial import ConvexHull

fig = plt.figure(figsize=(10, 8))
plt.scatter(group2.r_HD, group2.Gamma,
            c = normalized_group2_for_kmeans.c, alpha=0.6, s=80)
plt.scatter(original_cents[:, 0], original_cents[:, 2], c = colors, marker='^', s = 160)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter3d(x =group2.r_HD,
                           y=group2.r_HI,
                           z=group2.Gamma,
                           mode='markers',
                           marker=dict(
                               symbol=normalized_group2_for_kmeans.symbol,
                               size=10,
                               color=normalized_group2_for_kmeans.c,
                               opacity=0.8
                           )
                           ))
# fig.update_traces(marker_symbol=normalized_data_for_kmeans.loc[unchosen_patients_list].cluster)
# fig.update_traces(marker_color=normalized_data_for_kmeans.loc[unchosen_patients_list].c)

# fig = px.scatter_3d(data_for_kmeans.loc[unchosen_patients_list], x='r_HD', y='r_HI', z='Gamma',
#                     color=normalized_data_for_kmeans['c'], symbol=normalized_data_for_kmeans['cluster'])
# fig = px.scatter_3d(data_for_kmeans.loc[chosen_patients_list], x='r_HD', y='r_HI', z='Gamma',
#                     color=normalized_data_for_kmeans['c'], symbol=normalized_data_for_kmeans['cluster'])
fig2.update_xaxes(title=dict(text=r'$r_{HD}$'))
fig2.update_yaxes(title=dict(text=r'$r_{HI}$'))
# fig.update_layout(
#     xaxis_title=r'$r_{HD}$',
#     yaxis_title=r'$r_{HI}$'
# )
fig2.write_html('resistance_2_figure.html', auto_open=True)

#### Gaussian Mixture  model ####
normalized_group2_for_gm = pd.DataFrame({'y': list(group2.y),
                                           'r_HD': normalized_group2[:, 0],
                                           'r_HI': normalized_group2[:, 1],
                                           'Gamma': normalized_group2[:, 2]}, index=group2.index)
gm_model = GaussianMixture(n_components=k_num).fit(normalized_group2_for_gm[['r_HD', 'r_HI', 'Gamma']])
normalized_group2_for_gm['cluster'] = gm_model.predict(normalized_group2_for_gm[['r_HD', 'r_HI', 'Gamma']])
normalized_group2_for_gm['c'] = normalized_group2_for_gm.cluster.map({i: colors[i] for i in range(k_num)})
normalized_group2_for_gm['symbol'] = normalized_group2_for_gm.cluster.map({i: symbols[i] for i in range(k_num)})

fig3 = plt.figure(figsize=(10, 8))
plt.scatter(group2.r_HI, group2.Gamma,
            c = normalized_group2_for_gm.c, alpha=0.6, s=80)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.show()

