import sys
sys.path.append("..")

import numpy as np
import cv2
import os.path as osp
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
from loguru import logger
# import maxflow

from utils.common import test_dir_if_not_create

class KMCluster(object):
    """
    KMeans cluster
    Funcs:
        1. cluster the area centers
    """
    def __init__(self, save_path) -> None:
        """
        Args:
            center_list: list of cluster centers, 1~N
        """
        self.center_list = []
        self.save_path = save_path

    def load_center_list(self, center_list):
        """ load the pre-defined cluster center list
        """
        self.center_list = center_list

    def cluste_2d_points(self, points, show=False, name=""):
        """Cluster 2D points and select a candidate count from inertia drops.
        Args:
            points: np.ndarray[[u, v]...]
        Returns:
            label_num
            labels: list of labels
        """
        if len(self.center_list) == 0:
            logger.error("center_list is empty, please load the center_list first")
            return None, None
        
        inertia_list = []
        labels_list = []
        for cluster_num in self.center_list:
            kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init='auto').fit(points)
            labels = kmeans.labels_
            inertia = kmeans.inertia_
            inertia_list.append(inertia)
            labels_list.append(labels)
            if inertia < 1e-6:
                break
        
        # Score candidate cluster counts from adjacent inertia values.
        inertia_diff_front_list = []
        inertia_diff_back_list = []
        for i in range(1, len(inertia_list)-1):
            inertia_diff_front_list.append(inertia_list[i-1] - inertia_list[i])
            inertia_diff_back_list.append(inertia_list[i] - inertia_list[i-1])

    
        front_back_diff = np.array(inertia_diff_front_list) - np.array(inertia_diff_back_list)
        if show:
            plt.plot(self.center_list[1:-1], front_back_diff, 'bx-')
            plt.savefig(osp.join(self.save_path, name+"_elbow_diff.png"))
            plt.close()

        # get the biggest diff idx

        if len(front_back_diff) == 0:
            best_cluster_num_idx = 0
            best_cluster_num = self.center_list[best_cluster_num_idx]
        else:
            best_cluster_num_idx = np.argmax(front_back_diff) + 1
            best_cluster_num = self.center_list[best_cluster_num_idx]
            logger.debug("best_cluster_num: {}".format(best_cluster_num))

        # Plot inertia over the configured candidate counts.
        if show:
            plt.plot(self.center_list, inertia_list, 'bx-')
            plt.savefig(osp.join(self.save_path, name+"_elbow.png"))
            plt.close()

        best_labels = labels_list[best_cluster_num_idx]

        # plot the cluster result
        if show:
            plt.scatter(points[:, 0], points[:, 1], c=best_labels, s=50, cmap='viridis')
            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
            plt.savefig(osp.join(self.save_path, name+"_cluster.png"))
            plt.close()

        return best_cluster_num, best_labels

class AGViewer(object):
    """ Area Graph Visualization
    """
    def __init__(self, W, H, save_path) -> None:
        """
        """
        self.W = W
        self.H = H
        self.save_path = save_path

    def spring_layout_by_level(self, graph, AGNodes, level_num, node_dist=2):
        """
        """
        nodes_list = list(graph.nodes())
        fig_size_h = level_num*3
        # get the level of each node
        level_list = []
        for node in AGNodes:
            level_list.append(node.level)
        level_list = np.array(level_list)

        # get the nodes in each level
        nodes_in_level = defaultdict(list)
        for i in range(0,level_num+1):
            nodes_in_level[i] += np.where(level_list == i)[0].tolist()

        # get the max node number in all level
        max_node_num = 0
        for i in range(0, level_num+1):
            if len(nodes_in_level[i]) > max_node_num:
                max_node_num = len(nodes_in_level[i])
        
        fig_size_w = max_node_num*(node_dist+2)

        plt.rcParams['figure.figsize']= (int(fig_size_w), int(fig_size_h))



        # get the position of each node
        pos = {}
        for i in range(0, level_num+1):
            for j in range(len(nodes_in_level[i])):
                pos[int(nodes_in_level[i][j])] = np.array([j*(node_dist+2)+i%2, fig_size_h-i-1])

        return pos


    def draw_from_adjMat(self, adjMat, AGNodes, highlighted_idx_list=[], level_num=4, node_dist=2, name="", save=False):
        """ draw area graph from adjMat
        """
        G = nx.from_numpy_array(adjMat, create_using=nx.DiGraph)

        # draw graph whose edges with different weights are in different colors
        # pos = nx.spring_layout(G)
        # pos = nx.spectral_layout(G)
        pos = self.spring_layout_by_level(G, AGNodes, level_num=level_num, node_dist=node_dist)

        # get differnt edges
        e_neibour = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 1]
        e_father_son = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 2]

        # get node labels
        node_labels = {}
        for node in G.nodes():
            node_labels[node] = node

        highlight_nodes = []
        for i, node in enumerate(G.nodes()):
            if i in highlighted_idx_list:
                highlight_nodes.append(node)

        # draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='r')
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_size=3000, node_color='b')
        
        # draw node labels
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=50)
        # draw edges in e_neibour with directed=False
        nx.draw_networkx_edges(G, pos, edgelist=e_neibour, width=3, alpha=0.5, edge_color='b', style='dashed', arrows=False)
        # draw edges in e_father_son with directed=True
        nx.draw_networkx_edges(G, pos, edgelist=e_father_son, width=3, alpha=0.5, edge_color='g', arrows=True)

        if save:
            test_dir_if_not_create(self.save_path)
            plt.savefig(osp.join(self.save_path, f"{name}.png"))

        plt.close()
        
    def draw_single_node_area_in_img(self, img_, area_node, color=(0, 255, 0), name="", save=False):
        """
        Args:
            img: np.ndarray, shape=(H, W, 3)
            area_node: AGNode
                area: [u_min, u_max, v_min, v_max]
                idx: int
            color: tuple, (r, g, b)
        """
        img = copy.deepcopy(img_)
        area = area_node.area
        idx = area_node.idx

        # draw area rect in img
        cv2.rectangle(img, (area[0], area[2]), (area[1], area[3]), color, 2)

        # draw area idx in img
        cv2.putText(img, str(idx), ((area[0]+area[1])//2, (area[2]+area[3])//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        if save:
            cv2.imwrite(osp.join(self.save_path, f"{name}.jpg"), img)

        return img

    def draw_multi_nodes_areas_in_img(self, img, area_nodes, name="", save=False):
        """
        Args:
            img: np.ndarray, shape=(H, W, 3)
            area_nodes: list, [AGNode, AGNode, ...]
            name: str
            save: bool
        """
        img_ = copy.deepcopy(img)
        for area_node in area_nodes:
            color = np.random.randint(0, 255, size=3)
            color = tuple(color.tolist())
            img_ = self.draw_single_node_area_in_img(img_, area_node, color, name=name, save=save)
        
        if save:
            test_dir_if_not_create(self.save_path)
            cv2.imwrite(osp.join(self.save_path, f"{name}.jpg"), img_)

        return img_
