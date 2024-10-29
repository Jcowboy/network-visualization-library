import numpy as np
import graph_tool.all as gt
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import networkx as nx
import copy

#模糊社团可视化
def FCV_read_file(file_path):
    row = []
    col = []
    data = []
    with open(file_path,'r',encoding="utf-8") as f:
        for line in f:
            try:
                r,c = line.strip().split(' ')
            except:
                r,c = line.strip().split('	')
            row.append(int(r))
            col.append(int(c))
            data.append(1)
    a = max(max(row),max(col)) + 1
    return coo_matrix((data, (row, col)), shape=(a, a))

def FCV_get_departition(B):
    idx = np.argmax(B, axis=1)
    pos_node=[]
    for i in range(len(B)):
        pos_node.append(i)
    part=dict(zip(pos_node,idx))
    return part

# 定义计算半局部中心性的函数
def FCV_local_centrality(A):  # A为邻接矩阵，n为节点个数
    n = len(A)
    N = np.zeros(n)  # N为最近邻和次近邻个数和
    for i in np.arange(n):
        neibor = np.where(A[i, ]==1)  # 求最近邻节点下标
        for k in neibor[0][np.arange(len(neibor[0]))]:
            neibor2 = np.where(A[k, ]==1)  # 次近邻节点下标
            neibor = np.union1d(neibor, neibor2) # 合并得到邻居节点和次邻居节点和当前节点下标
        N[i] = len(neibor)-1  # 去除当前节点
    Q = np.zeros(n)
    for i in np.arange(n):
        Q[i] = sum(N[np.where(A[i, ]==1)])
    # 计算局部中心性
    LC = np.zeros(n)
    for i in np.arange(n):
        LC[i] = sum(Q[np.where(A[i, ]==1)])
    return {i:LC[i] for i in range(len(LC))}

def FCV_get_center_points(G_matrix,B): #获取中心节点                                    加B
    part = FCV_get_departition(B)                 
    L_C = FCV_local_centrality(G_matrix)
    #print("节点局部中心性:\n",L_C)
    result = {}
    for key,value in part.items():
        if value not in result:
            result[value] = {'points':[],'center_point':None}
        result[value]['points'].append(key)
        #result[value]['points']=community_list1
        
        if result[value]['center_point'] == None:
            result[value]['center_point'] = key
        else:
            if L_C[key] > L_C[result[value]['center_point']]:
                result[value]['center_point'] = key
    return result

def FCV_init_pos(G):#初始化节点坐标
    np.random.seed(1)
    nodes_num = G.shape[0]
    G = G.tolil()
    G = G.astype('float')
    pos = np.asarray(np.random.rand(nodes_num, 2), dtype=G.dtype)
    return pos


def FCV_pos_updata_1(G,Graph,pos,k,step_length):#根据节点之间的斥力和相邻节点的引力更新节点坐标       加G   
    displacement = np.zeros((2, pos.shape[0]))
    for i in range(G.shape[0]):
        delta = (pos[i] - pos).T
        distance = np.sqrt((delta ** 2).sum(axis=0))
        distance = np.where(distance < 0.01, 0.01, distance)
        Ai = np.asarray(Graph[i,:])
        displacement[:, i] += \
            (delta * (k * k / distance**2 - Ai * distance / k)).sum(axis=1)
    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.01, 0.01, length)
    delta_pos = (displacement.T * step_length/np.mean(length) )
    pos += delta_pos
    return pos

def FCV_pos_updata_2(pos,step_length,partitionWithCenters,g_cr):#社区中心节点之间的斥力
    displacement = np.zeros((2, pos.shape[0]))
    for value in partitionWithCenters.values():
        center_point = value['center_point']
        for value1 in partitionWithCenters.values():
            if value != value1:
                center_point1 = value1['center_point']
                # center.append(center_point)
                delta = (pos[center_point]-pos[center_point1]).T
                distance = np.sqrt((delta ** 2).sum(axis=0))
                if distance < 0.15:
                    distance = np.where(distance < 0.1, 0.1, distance)
                    displacement[:, center_point] += delta *((len(value['points']))*(len(value1['points'])))/(distance*pos.shape[0]**2)
                else:
                    distance = np.where(distance < 0.1, 0.1, distance)
                    displacement[:, center_point] -= delta * ((len(value['points'])) * (len(value1['points']))) / (
                                distance * pos.shape[0] ** 2)
    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.1, 0.1, length)
    delta_pos = (displacement.T * step_length/np.mean(length) )
    pos += g_cr*delta_pos
    return pos

def FCV_pos_updata_3(B,pos,step_length,partitionWithCenters,g_ca):#社区内所有节点对社团中心的引力      #加B 
    displacement = np.zeros((pos.shape[0], 2, pos.shape[0]))
    delta=np.zeros((4,2))
    distance=np.zeros((4,1))
    length=np.zeros((34,1,34))
    delta_pos=np.zeros((34,34,2))
    delta_pos1=np.zeros((34,2))
    for key,value in partitionWithCenters.items():
        center_point = value['center_point']
        for j in value['points']:
            if j!=center_point:
                for i in range(4):
                    delta[i]=(pos[j]-pos[partitionWithCenters[i]['center_point']]).T
                    distance[i] = np.sqrt((delta[i] ** 2).sum(axis=0))
                    distance[i] = np.where(distance[i] < 0.01, 0.3, distance[i])
                    displacement[j][:,j] -= delta[i]*g_ca*distance[i]*len(value['points'])*B[j][i]
    for i in range(pos.shape[0]):
        length[i] = np.sqrt((displacement[i] ** 2).sum(axis=0))
        length[i] = np.where(length[i] < 0.001, 0.001, length[i])
        delta_pos[i] = (displacement[i].T * step_length/np.mean(length[i]))
        delta_pos1[i]=delta_pos[i,i,:]
    pos += delta_pos1
    return pos

def FCV_evolution(B,G_matrix,partitionWithCenters,G,t=None,C = 1,g_cr = 0.4,g_ca=5.0,iterations=600, threshold=1e-5):           #加B,G_matrix、partitionWithCenters
    # 初始化节点坐标
    pos = FCV_init_pos(G)
    #得到k
    nodes_num = G.shape[0]
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1]))*0.1
    dt = t / float(iterations + 1)
    k = np.sqrt(1.0 / nodes_num)
    for iter in range(iterations):
        pos_1 = FCV_pos_updata_1(G,G_matrix,pos.copy(),k,t)
        pos_2 = FCV_pos_updata_2(pos_1.copy(),  t, partitionWithCenters,g_cr)
        pos_3 = FCV_pos_updata_3(B,pos_2.copy(),  t, partitionWithCenters,g_ca)
        err = np.linalg.norm(pos_3-pos) / float(nodes_num)
        pos = pos_3.copy()
        if err < threshold or t < 0:
            break
        t -= dt
    return pos

def FCV_getG(pos1,filepath):
    G=gt.Graph(directed=False)
    G.add_vertex(pos1.shape[0])
    edges=[]
    for line in open(filepath): 
        n1,n2=line.strip('\n').split(' ')
        edges.append((int(n1),int(n2)))
    G.add_edge_list(edges)
    
    return G

def FCV_getpos(pos1,draw_G):
    pos=gt.fruchterman_reingold_layout(draw_G)
    k=0
    k=int(k)
    for i in pos:
        i[0]=pos1[k][0]
        i[1]=pos1[k][1]
        k+=1
    return pos

def fuzzy_community_visualization(filepath,B):
    G = FCV_read_file(filepath)
    G_matrix = G.toarray()
    Gn_karate = nx.read_edgelist(filepath,nodetype=int)
    G1 = Gn_karate.to_undirected()
    B=np.array(B)
    partitionWithCenters = FCV_get_center_points(G_matrix,B)
    pos1=FCV_evolution(B,G_matrix,partitionWithCenters,G, t=None, C=1, g_cr=1.1, g_ca=0.03, iterations=100, threshold=1e-5) #karate参数
    draw_G=FCV_getG(pos1,filepath)
    pos=FCV_getpos(pos1,draw_G)
    vp=draw_G.new_vertex_property("vector<double>")
    size = []
    for i in B:
        size.append(np.count_nonzero(i!=0))
    for i in range(pos1.shape[0]):
        vp[draw_G.vertex(i)]=B[i]
    gt.graph_draw(draw_G,pos=pos,vertex_size=15*(B.shape[1]+size[i])/B.shape[1],vertex_text=draw_G.vertex_index,vertex_font_size=10,vertex_text_color='black',vertex_shape='pie',vertex_pie_fractions=vp,output='overlapping模糊1.png',inline=True)






#离散重叠社团可视化
def DOCV_read_file(file_path):
    row = []
    col = []
    data = []
    with open(file_path,'r',encoding="utf-8") as f:
        for line in f:
            try:
                r,c = line.strip().split(' ')
            except:
                r,c = line.strip().split('	')
            row.append(int(r))
            col.append(int(c))
            data.append(1)
    a = max(max(row),max(col)) + 1
    return coo_matrix((data, (row, col)), shape=(a, a))

def DOCV_get_numpy(community_list,G1):                                                 #加G1
    result={}
    result=result.fromkeys(range(len(G1)))
    for i in range(len(G1)):
        result[i]=[]
    for i in range(len(community_list)):
        for j in community_list[i]:
            result[j].append(i)                
    B=np.zeros((len(G1),len(community_list)))
    for key,value in result.items():
        if len(result[key])==1:
            B[key][result[key]]=1
        else:
            B[key][result[key]]=1/len(result[key])
            if (np.sum(B[key]))!=1:
                B[key][result[len(result[key])]]=1-(len(result[key])-1)/len(result[key])
    return B

def DOCV_get_max(community_list,G1):                                                     #加G1
    result={}
    result=result.fromkeys(range(len(G1)))
    for i in range(len(G1)):
        result[i]=[]
    for i in range(len(community_list)):
        for j in community_list[i]:
            result[j].append(i)
    result1=copy.deepcopy(result)
    community_list1=[]
    v=[]
    for i in range(4):
        community_list1.append([])
    for key,value in result.items():
        if len(result[key])==1:
            community_list1[value[0]].append(key)
        else:
            v.append(key)
    for i in v:
        for j,k in enumerate(result[i]):
            result[i][j]=len(community_list1[result[i][j]])
            u=(result[i]).index(min(result[i]))
        community_list1[result1[i][u]].append(i)
    return community_list1

def DOCV_get_departition(community_list,G1):                                     # 加G1
    community_list1=DOCV_get_max(community_list,G1)
    k=[]
    u=[]
    for i in range(len(community_list1)):
        for j in community_list1[i]:
            k.append(j)
            u.append(i)
    part=dict(zip(k,u))
    return part

# 定义计算半局部中心性的函数
def DOCV_local_centrality(A):  # A为邻接矩阵，n为节点个数
    n = len(A)
    N = np.zeros(n)  # N为最近邻和次近邻个数和
    for i in np.arange(n):
        neibor = np.where(A[i, ]==1)  # 求最近邻节点下标
        for k in neibor[0][np.arange(len(neibor[0]))]:
            neibor2 = np.where(A[k, ]==1)  # 次近邻节点下标
            neibor = np.union1d(neibor, neibor2) # 合并得到邻居节点和次邻居节点和当前节点下标
        N[i] = len(neibor)-1  # 去除当前节点
    Q = np.zeros(n)
    for i in np.arange(n):
        Q[i] = sum(N[np.where(A[i, ]==1)])
    # 计算局部中心性
    LC = np.zeros(n)
    for i in np.arange(n):
        LC[i] = sum(Q[np.where(A[i, ]==1)])
    return {i:LC[i] for i in range(len(LC))}

def DOCV_get_center_points(G_matrix,community_list1,G1): #获取中心节点                                    #加community_list1,G1
    part = DOCV_get_departition(community_list1,G1)
    L_C = DOCV_local_centrality(G_matrix)
    #print("节点局部中心性:\n",L_C)
    result = {}
    for key,value in part.items():
        if value not in result:
            result[value] = {'points':[],'center_point':None}
        result[value]['points'].append(key)
        #result[value]['points']=community_list1
        
        if result[value]['center_point'] == None:
            result[value]['center_point'] = key
        else:
            if L_C[key] > L_C[result[value]['center_point']]:
                result[value]['center_point'] = key
    return result

def DOCV_init_pos(G):#初始化节点坐标
    np.random.seed(1)
    nodes_num = G.shape[0]
    G = G.tolil()
    G = G.astype('float')
    pos = np.asarray(np.random.rand(nodes_num, 2), dtype=G.dtype)
    return pos

def DOCV_pos_updata_1(G,Graph,pos,k,step_length):#根据节点之间的斥力和相邻节点的引力更新节点坐标           #加G
    displacement = np.zeros((2, pos.shape[0]))
    for i in range(G.shape[0]):
        delta = (pos[i] - pos).T
        distance = np.sqrt((delta ** 2).sum(axis=0))
        distance = np.where(distance < 0.01, 0.01, distance)
        Ai = np.asarray(Graph[i,:])
        displacement[:, i] += \
            (delta * (k * k / distance**2 - Ai * distance / k)).sum(axis=1)
    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.01, 0.01, length)
    delta_pos = (displacement.T * step_length/np.mean(length) )
    pos += delta_pos
    return pos

def DOCV_pos_updata_2(pos,step_length,partitionWithCenters,g_cr):#社区中心节点之间的斥力
    displacement = np.zeros((2, pos.shape[0]))
    for value in partitionWithCenters.values():
        center_point = value['center_point']
        for value1 in partitionWithCenters.values():
            if value != value1:
                center_point1 = value1['center_point']
                # center.append(center_point)
                delta = (pos[center_point]-pos[center_point1]).T
                distance = np.sqrt((delta ** 2).sum(axis=0))
                if distance < 0.15:
                    distance = np.where(distance < 0.1, 0.1, distance)
                    displacement[:, center_point] += delta *((len(value['points']))*(len(value1['points'])))/(distance*pos.shape[0]**2)
                else:
                    distance = np.where(distance < 0.1, 0.1, distance)
                    displacement[:, center_point] -= delta * ((len(value['points'])) * (len(value1['points']))) / (
                                distance * pos.shape[0] ** 2)
    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.1, 0.1, length)
    delta_pos = (displacement.T * step_length/np.mean(length) )
    pos += g_cr*delta_pos
    return pos

def DOCV_pos_updata_3(community_list,B,pos,step_length,partitionWithCenters,g_ca):#社区内所有节点对社团中心的引力      加community_list、B
    displacement = np.zeros((pos.shape[0], 2, pos.shape[0]))
    delta=np.zeros((len(community_list),2))
    distance=np.zeros((len(community_list),1))
    length=np.zeros((pos.shape[0],1,pos.shape[0]))
    delta_pos=np.zeros((pos.shape[0],pos.shape[0],2))
    delta_pos1=np.zeros((pos.shape[0],2))
    for key,value in partitionWithCenters.items():
        center_point = value['center_point']
        for j in value['points']:
            if j!=center_point:
                for i in range(len(community_list)):
                    delta[i]=(pos[j]-pos[partitionWithCenters[i]['center_point']]).T
                    distance[i] = np.sqrt((delta[i] ** 2).sum(axis=0))
                    distance[i] = np.where(distance[i] < 0.01, 0.3, distance[i])
                    displacement[j][:,j] -= delta[i]*g_ca*distance[i]*len(value['points'])*B[j][i]
    for i in range(pos.shape[0]):
        length[i] = np.sqrt((displacement[i] ** 2).sum(axis=0))
        length[i] = np.where(length[i] < 0.001, 0.001, length[i])
        delta_pos[i] = (displacement[i].T * step_length/np.mean(length[i]))
        delta_pos1[i]=delta_pos[i,i,:]
    pos += delta_pos1
    return pos

def DOCV_evolution(community_list,B,G_matrix,partitionWithCenters,G,t=None,C = 1,g_cr = 0.4,g_ca=5.0,iterations=600, threshold=1e-5):                        #加community_list,B,G_matrix\partitionWithCenters
    # 初始化节点坐标
    pos = DOCV_init_pos(G)
    #得到k
    nodes_num = G.shape[0]
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1]))*0.1
    dt = t / float(iterations + 1)
    k = np.sqrt(1.0 / nodes_num)
    for iter in range(iterations):
        pos_1 = DOCV_pos_updata_1(G,G_matrix,pos.copy(),k,t)
        pos_2 = DOCV_pos_updata_2(pos_1.copy(),  t, partitionWithCenters,g_cr)
        pos_3 = DOCV_pos_updata_3(community_list,B,pos_2.copy(),  t, partitionWithCenters,g_ca)
        err = np.linalg.norm(pos_3-pos) / float(nodes_num)
        pos = pos_3.copy()
        if err < threshold or t < 0:
            break
        t -= dt
    return pos

def DOCV_getG(pos1,filepath):
    G=gt.Graph(directed=False)
    G.add_vertex(pos1.shape[0])
    edges=[]
    for line in open(filepath): 
        n1,n2=line.strip('\n').split(' ')
        edges.append((int(n1),int(n2)))
    G.add_edge_list(edges)
    return G

def DOCV_getpos(pos1,draw_G):
    pos=gt.fruchterman_reingold_layout(draw_G)
    k=0
    k=int(k)
    for i in pos:
        i[0]=pos1[k][0]
        i[1]=pos1[k][1]
        k+=1
    return pos

def discrete_community_visualization(filepath, community_list):
    G = DOCV_read_file(filepath)
    G_matrix = G.toarray()
    Gn_karate = nx.read_edgelist(filepath,nodetype=int)
    G1 = Gn_karate.to_undirected()
    community_list1=DOCV_get_max(community_list,G1)
    B=DOCV_get_numpy(community_list,G1)
    partitionWithCenters = DOCV_get_center_points(G_matrix,community_list1,G1)
    pos1=DOCV_evolution(community_list,B,G_matrix,partitionWithCenters,G, t=None, C=1, g_cr=0.7, g_ca=0.015, iterations=110, threshold=1e-5) #karate参数
    draw_G=DOCV_getG(pos1,filepath)
    pos=DOCV_getpos(pos1,draw_G)
    vp=draw_G.new_vertex_property("vector<double>")
    for i in range(pos1.shape[0]):
        vp[draw_G.vertex(i)]=B[i]
    gt.graph_draw(draw_G,pos=pos,vertex_size=21,vertex_text=draw_G.vertex_index,vertex_font_size=10,vertex_text_color='black',vertex_shape='pie',vertex_pie_fractions=vp,output='overlapping离散.svg',inline=True)  
    







#固定节点的可视化
def VOFN_read_file(file_path):
    row = []
    col = []
    data = []
    with open(file_path,'r',encoding="utf-8") as f:
        for line in f:
            try:
                r,c = line.strip().split(' ')
            except:
                r,c = line.strip().split('	')
            row.append(int(r))
            col.append(int(c))
            data.append(1)
    a = max(max(row),max(col)) + 1
    return coo_matrix((data, (row, col)), shape=(a, a))

def VOFN_get_departition(community_list):
    k=[]
    u=[]
    for i in range(len(community_list)):
        for j in community_list[i]:
            k.append(j)
            u.append(i)
    part=dict(zip(k,u))
    return part
# 定义计算半局部中心性的函数
def VOFN_local_centrality(A):  # A为邻接矩阵，n为节点个数
    n = len(A)
    N = np.zeros(n)  # N为最近邻和次近邻个数和
    for i in np.arange(n):
        neibor = np.where(A[i, ]==1)  # 求最近邻节点下标
        for k in neibor[0][np.arange(len(neibor[0]))]:
            neibor2 = np.where(A[k, ]==1)  # 次近邻节点下标
            neibor = np.union1d(neibor, neibor2) # 合并得到邻居节点和次邻居节点和当前节点下标
        N[i] = len(neibor)-1  # 去除当前节点
    Q = np.zeros(n)
    for i in np.arange(n):
        Q[i] = sum(N[np.where(A[i, ]==1)])
    # 计算局部中心性
    LC = np.zeros(n)
    for i in np.arange(n):
        LC[i] = sum(Q[np.where(A[i, ]==1)])
    return {i:LC[i] for i in range(len(LC))}

def VOFN_get_center_points(community_list,G_matrix):                                   #获取中心节点    加community_list
    part = VOFN_get_departition(community_list)
    L_C = VOFN_local_centrality(G_matrix)
    result = {}
    for key,value in part.items():
        if value not in result:
            result[value] = {'points':[],'center_point':None}
        result[value]['points'].append(key)
        if result[value]['center_point'] == None:
            result[value]['center_point'] = key
        else:
            if L_C[key] > L_C[result[value]['center_point']]:
                result[value]['center_point'] = key
    return result

def VOFN_init_pos(G):#初始化节点坐标
    np.random.seed(1)
    nodes_num = G.shape[0]
    G = G.tolil()
    G = G.astype('float')
    pos = np.asarray(np.random.rand(nodes_num, 2), dtype=G.dtype)
    return pos

def VOFN_pos_updata_1(G,Graph,pos,k,step_length):#根据节点之间的斥力和相邻节点的引力更新节点坐标           加G
    displacement = np.zeros((2, pos.shape[0]))
    for i in range(G.shape[0]):
        delta = (pos[i] - pos).T
        distance = np.sqrt((delta ** 2).sum(axis=0))
        distance = np.where(distance < 0.01, 0.01, distance)
        Ai = np.asarray(Graph[i,:])
        #             print('Ai', Ai)
        displacement[:, i] += \
            (delta * (k * k / distance**2 - Ai * distance / k)).sum(axis=1)
        # update positions
    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.01, 0.01, length)
    delta_pos = (displacement.T * step_length/np.mean(length) )
    pos += delta_pos
    return pos

def VOFN_pos_updata_2(pos,step_length,partitionWithCenters,g_cr):#社区中心节点之间的斥力
    displacement = np.zeros((2, pos.shape[0]))
    for value in partitionWithCenters.values():
        center_point = value['center_point']
        for value1 in partitionWithCenters.values():
            if value != value1:
                center_point1 = value1['center_point']
                # center.append(center_point)
                delta = (pos[center_point]-pos[center_point1]).T
                distance = np.sqrt((delta ** 2).sum(axis=0))
                if distance < 0.15:
                    distance = np.where(distance < 0.1, 0.1, distance)
                    displacement[:, center_point] += delta *((len(value['points']))*(len(value1['points'])))/(distance*pos.shape[0]**2)
                else:
                    distance = np.where(distance < 0.1, 0.1, distance)
                    displacement[:, center_point] -= delta * ((len(value['points'])) * (len(value1['points']))) / (
                                distance * pos.shape[0] ** 2)
    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.1, 0.1, length)
    delta_pos = (displacement.T * step_length/np.mean(length) )
    pos += g_cr*delta_pos
    return pos

def VOFN_pos_updata_3(pos,step_length,partitionWithCenters,g_ca):#社区中心节点之间的引力
    displacement = np.zeros((2, pos.shape[0]))
    for value in partitionWithCenters.values():
        center_point = value['center_point']
        for j in value['points']:
            if j!=center_point:
                delta = (pos[j] - pos[center_point]).T
                distance = np.sqrt((delta ** 2).sum(axis=0))
                distance = np.where(distance < 0.01, 0.3, distance)
                displacement[:, j] -= delta*g_ca*distance*len(value['points'])
    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.001, 0.001, length)
    delta_pos = (displacement.T * step_length/np.mean(length) )

    pos += delta_pos
    return pos

def VOFN_evolution(G_matrix,partitionWithCenters,G,t=None,C = 1,g_cr = 0.4,g_ca=5.0,iterations=600, threshold=1e-5):                   #加G_matrix，partitionWithCenters
    # 初始化节点坐标
    pos = VOFN_init_pos(G)
    #得到k
    nodes_num = G.shape[0]
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1]))*0.1
    dt = t / float(iterations + 1)
    k = np.sqrt(1.0 / nodes_num)
    for iter in range(iterations):
        pos_1 = VOFN_pos_updata_1(G,G_matrix,pos.copy(),k,t)
        pos_2 = VOFN_pos_updata_2(pos_1.copy(),  t, partitionWithCenters,g_cr)
        pos_3 = VOFN_pos_updata_3(pos_2.copy(),  t, partitionWithCenters,g_ca)
        err = np.linalg.norm(pos_3-pos) / float(nodes_num)
        pos = pos_3.copy()
        if err < threshold or t < 0:
            break
        t -= dt
    return pos

def Visualization_of_fixed_nodes(filepath,community_list,colors,options):           #例如colors = ['#eb8f90', '#ffb471', '#adbed2', '#12406a']   options = {'font_family': 'serif', 'font_size': '8', 'font_color': '#ffffff'}      colors列表中颜色的数量根据社区的数量而定
    G = VOFN_read_file(filepath)
    G_matrix = G.toarray()
    #调整节点位置
    partitionWithCenters = VOFN_get_center_points(community_list,G_matrix)
    pos = VOFN_evolution(G_matrix,partitionWithCenters,G, t=None, C=1, g_cr=0.7, g_ca=0.8, iterations=110, threshold=1e-5)  # karate参数
    #可视化节点并保存图片
    Gn_karate = nx.read_edgelist(filepath, nodetype=int)
    G1 = Gn_karate.to_undirected()
    nx.draw(G1, pos, edge_color="gray", with_labels=True, node_size=260, **options)
    for i in range(len(community_list)):
        nx.draw_networkx_nodes(G1, pos, nodelist=community_list[i], node_size=300, node_color=colors[i], label=True)
    plt.savefig("固定节点位置2.svg")




#加权网络可视化

def WUNCV_read_file(file_path):
    row = []
    col = []
    data = []
    with open(file_path,'r',encoding="utf-8") as f:
        for line in f:
            try:
                r,c,u= line.strip().split(' ')
            except:
                r,c,u= line.strip().split(' ')
            row.append(int(r))
            col.append(int(c))
            data.append(1)
    a = max(max(row),max(col)) + 1
    return coo_matrix((data, (row, col)), shape=(a, a))

def WUNCV_get_departition(community_list):
    k=[]
    u=[]
    for i in range(len(community_list)):
        for j in community_list[i]:
            k.append(j)
            u.append(i)
    part=dict(zip(k,u))
    return part

# 定义计算半局部中心性的函数
def WUNCV_local_centrality(A):  # A为邻接矩阵，n为节点个数
    n = len(A)
    N = np.zeros(n)  # N为最近邻和次近邻个数和
    for i in np.arange(n):
        neibor = np.where(A[i, ]==1)  # 求最近邻节点下标
        for k in neibor[0][np.arange(len(neibor[0]))]:
            neibor2 = np.where(A[k, ]==1)  # 次近邻节点下标
            neibor = np.union1d(neibor, neibor2) # 合并得到邻居节点和次邻居节点和当前节点下标
        N[i] = len(neibor)-1  # 去除当前节点
    Q = np.zeros(n)
    for i in np.arange(n):
        Q[i] = sum(N[np.where(A[i, ]==1)])
    # 计算局部中心性
    LC = np.zeros(n)
    for i in np.arange(n):
        LC[i] = sum(Q[np.where(A[i, ]==1)])
    return {i:LC[i] for i in range(len(LC))}

def WUNCV_get_center_points(community_list,G_matrix): #获取中心节点                               加community_list
    part = WUNCV_get_departition(community_list)
    L_C = WUNCV_local_centrality(G_matrix)
    result = {}
    for key,value in part.items():
        if value not in result:
            result[value] = {'points':[],'center_point':None}
        result[value]['points'].append(key)
        if result[value]['center_point'] == None:
            result[value]['center_point'] = key
        else:
            if L_C[key] > L_C[result[value]['center_point']]:
                result[value]['center_point'] = key
    return result

def WUNCV_init_pos(G):#初始化节点坐标
    np.random.seed(1)
    nodes_num = G.shape[0]
    G = G.tolil()
    G = G.astype('float')
    pos = np.asarray(np.random.rand(nodes_num, 2), dtype=G.dtype)
    return pos

def WUNCV_pos_updata_1(G,Graph,pos,k,step_length):#根据节点之间的斥力和相邻节点的引力更新节点坐标                 加G
    displacement = np.zeros((2, pos.shape[0]))
    for i in range(G.shape[0]):
        delta = (pos[i] - pos).T
        distance = np.sqrt((delta ** 2).sum(axis=0))
        distance = np.where(distance < 0.01, 0.01, distance)
        Ai = np.asarray(Graph[i,:])
        #             print('Ai', Ai)
        displacement[:, i] += \
            (delta * (k * k / distance**2 - Ai * distance / k)).sum(axis=1)
        # update positions
    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.01, 0.01, length)
    delta_pos = (displacement.T * step_length/np.mean(length) )
    pos += delta_pos
    return pos

def WUNCV_pos_updata_2(pos,step_length,partitionWithCenters,g_cr):#社区中心节点之间的斥力
    displacement = np.zeros((2, pos.shape[0]))
    for value in partitionWithCenters.values():
        center_point = value['center_point']
        for value1 in partitionWithCenters.values():
            if value != value1:
                center_point1 = value1['center_point']
                # center.append(center_point)
                delta = (pos[center_point]-pos[center_point1]).T
                distance = np.sqrt((delta ** 2).sum(axis=0))
                if distance < 0.15:
                    distance = np.where(distance < 0.1, 0.1, distance)
                    displacement[:, center_point] += delta *((len(value['points']))*(len(value1['points'])))/(distance*pos.shape[0]**2)
                else:
                    distance = np.where(distance < 0.1, 0.1, distance)
                    displacement[:, center_point] -= delta * ((len(value['points'])) * (len(value1['points']))) / (
                                distance * pos.shape[0] ** 2)
    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.1, 0.1, length)
    delta_pos = (displacement.T * step_length/np.mean(length) )
    pos += g_cr*delta_pos
    return pos

def WUNCV_pos_updata_3(pos,step_length,partitionWithCenters,g_ca):#社区中心节点之间的引力
    displacement = np.zeros((2, pos.shape[0]))
    for value in partitionWithCenters.values():
        center_point = value['center_point']
        for j in value['points']:
            if j!=center_point:
                delta = (pos[j] - pos[center_point]).T
                distance = np.sqrt((delta ** 2).sum(axis=0))
                distance = np.where(distance < 0.01, 0.3, distance)
                displacement[:, j] -= delta*g_ca*distance*len(value['points'])
    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.001, 0.001, length)
    delta_pos = (displacement.T * step_length/np.mean(length) )

    pos += delta_pos
    return pos

def WUNCV_evolution(G_matrix,partitionWithCenters,G,t=None,C = 1,g_cr = 0.4,g_ca=5.0,iterations=600, threshold=1e-5):            #加G_matrix,partitionWithCenters,
    # 初始化节点坐标
    pos = WUNCV_init_pos(G)
    #得到k
    nodes_num = G.shape[0]
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1]))*0.1
    dt = t / float(iterations + 1)
    k = np.sqrt(1.0 / nodes_num)
    for iter in range(iterations):
        pos_1 = WUNCV_pos_updata_1(G,G_matrix,pos.copy(),k,t)
        pos_2 = WUNCV_pos_updata_2(pos_1.copy(),  t, partitionWithCenters,g_cr)
        pos_3 = WUNCV_pos_updata_3(pos_2.copy(),  t, partitionWithCenters,g_ca)
        err = np.linalg.norm(pos_3-pos) / float(nodes_num)
        pos = pos_3.copy()
        if err < threshold or t < 0:
            break
        t -= dt
    return pos

#得到权重值
def WUNCV_getweight(filepath):
    weight=[]
    for line in open(filepath):
        n1,n2,n3=line.split(' ')
        weight.append(float(n3))
    return weight

#连边字典
def WUNCV_getdict(filepath,G1):
    edgeindex=[k+1 for k in range(len(G1.edges()))]
    edge=[]
    for line in open(filepath):
        n1,n2,n3=line.split(' ')
        edge.append([(int(n1),int(n2))])
    edgedict=dict(zip(edgeindex,edge))
    return edgedict

def WUNCV(filepath,community_list,colors,options):                                                   #weighted undirected network community visualization
    G = WUNCV_read_file(filepath)
    G_matrix = G.toarray()
    partitionWithCenters = WUNCV_get_center_points(community_list,G_matrix)
    pos = WUNCV_evolution(G_matrix,partitionWithCenters,G, t=None, C=1, g_cr=0.5, g_ca=1.1, iterations=120, threshold=1e-5)

    Gn_karate = nx.read_weighted_edgelist(filepath, nodetype=int)
    G1 = Gn_karate.to_undirected()
    plt.figure(figsize=(7, 7))
    nx.draw(G1, pos, edge_color="gray", with_labels=True, node_size=250, edgelist=[], **options)
    weight = WUNCV_getweight(filepath)
    edgedict = WUNCV_getdict(filepath, G1)
    for i in range(len(community_list)):
        nx.draw_networkx_nodes(G1, pos, nodelist=community_list[i], node_size=250, node_color=colors[i], label=True)
    for i in range(len(edgedict)):
        nx.draw_networkx_edges(G1, pos, edgelist=edgedict[i + 1], width=weight[i] / 5, alpha=0.7)
    plt.savefig("加权.svg", dpi=600)


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from scipy.linalg import eigh


# 符号网络可视化

def S_FR_read_file(file_path):
    row = []
    col = []
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            # split the line by comma and strip whitespace
            node1, node2, sign = line.split(" ")
            node1 = int(node1.strip())
            node2 = int(node2.strip())
            sign = int(sign.strip())
            # append the sign to data list as +1 or -1
            if sign == 1:
                data.append(1)
            elif sign == -1:
                data.append(-1)
            else:
                raise ValueError("Invalid sign")
            # append the nodes to row and col lists
            row.append(node1)
            col.append(node2)
        # create a coo_matrix from data, row and col lists
        a = max(max(row), max(col)) + 1
        matrix = sp.coo_matrix((data, (row, col)), shape=(a, a))
        # return the matrix
        return matrix


def S_FR_init_pos(G):  # 初始化节点坐标
    np.random.seed(1)
    nodes_num = G.shape[0]
    G = G.tolil()
    G = G.astype('float')
    pos = np.asarray(np.random.rand(nodes_num, 2), dtype=G.dtype)
    return pos


def S_FR_coefficient(G):
    # 计算网络结构参数
    # degree_max = max(dict(G.degree()).values())  # 最大节点的度数
    # degree_min = min(dict(G.degree()).values())  # 最小节点的度数
    diameter = nx.diameter(G)  # 网络的最远节点之间的距离
    avg_distance = nx.average_shortest_path_length(G)  # 网络中所有节点之间的平均距
    # repulsion_scale = degree_max / diameter
    # repulsion_scale = degree_max / (degree_min * avg_distance) * diameter
    # attraction_scale = degree_min / avg_distance
    # return repulsion_scale,attraction_scale
    return diameter, avg_distance


def S_FR_degree_i(A):
    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            A[j][i] = A[i][j]
        # 获取矩阵的行数和列数
    num_rows, num_cols = A.shape
    # 初始化一个二维矩阵，用于存储每行正数和负数的个数
    counts = np.zeros((num_rows, 2))

    # 遍历矩阵的每一行
    for i in range(num_rows):
        # 统计正数的个数
        counts[i, 0] = np.sum(A[i] > 0)
        # 统计负数的个数
        counts[i, 1] = np.sum(A[i] < 0)
    return counts


def S_FR_weighted(
        A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-5, dim=2, weight=None, seed=None, g_ca=2, g_cr=1.5
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    try:
        nnodes, _ = A.shape
    except AttributeError as err:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from err
    if pos is None:
        # random initial positions
        pos = seed.random(size=(nnodes, dim))
        # pos = np.asarray(B, dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos_array = np.array(list(pos.values()))  # Convert dictionary values to array
        pos_array = pos_array.astype(np.float64)
        pos = pos_array.astype(A.dtype)
    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    dt = t / float(iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    for iteration in range(iterations):
        # 计算pos中每个元素与其他元素之间的差值,其中第i行第j列的元素表示x[i]-x[j]的差值
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=-1)  # 用于计算向量或矩阵之间的距离，以评估它们之间的相似度或差异性
        # 其中第`i`行第`j`列的元素表示`x`的第`i`行第`j`列的向量的范数。
        np.clip(distance, 0.01, None, out=distance)
        Fa = A * distance / k
        Fr = k * k / distance ** 2
        # # degree_list=degree_i(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] < 0:
                    Fr[i, j] *= 50
                    Fr[j, i] = Fr[i, j]
                    # iii= degree_list[i][1]+degree_list[i][0]
                    # Fr[i, j] *= degree_list[i][1] / g_cr
                    # Fr[i, j] *= iii / g_cr
                elif A[i, j] > 0:
                    Fa[i, j] *= 2
                    # Fa[j, i] = Fa[i, j]
                    # ii=degree_list[i][1]+degree_list[i][0]
                    # Fa *=0.75*degree_list[i][0]/g_ca
                else:
                    j += 1
        displacement = np.einsum("ijk,ij->ik", delta, Fr - Fa)
        # displacement = np.einsum("ijk,ij->ik", delta, (k * k / distance ** 2 - A * distance / k))
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        qa = t / length
        delta_pos = np.einsum("ij,i->ij", displacement, qa)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # cool temperature
        t -= dt
        if (np.linalg.norm(delta_pos) / nnodes) < threshold or t < 0:
            break
    return pos


# 得到权重值
def S_FR_getweight(filepath):
    weight = []
    for line in open(filepath):
        n1, n2, n3 = line.split(' ')
        weight.append(float(n3))
    return weight


# 连边字典
def S_FR_getdict(filepath, G1):
    edgeindex = [k + 1 for k in range(len(G1.edges()))]
    edge = []
    for line in open(filepath):
        n1, n2, n3 = line.split(' ')
        edge.append([(int(n1), int(n2))])
    edgedict = dict(zip(edgeindex, edge))
    return edgedict


def signed_network_visualization(filepath, options,
                                 node_size=500):  # options = {'font_family': 'serif', 'font_size': '8', 'font_color': '#ffffff'}

    G = S_FR_read_file(filepath)
    G_matrix = G.toarray()
    Gn_RD = nx.read_weighted_edgelist("RD_0.txt", nodetype=int)
    G1 = Gn_RD.to_undirected()
    edgedict = S_FR_getdict(filepath, G1)
    weight = S_FR_getweight(filepath)
    seed = np.random.RandomState(0)
    # pos1 = init_pos(G)
    g_cr, g_ca = S_FR_coefficient(Gn_RD)

    pos = S_FR_weighted(G_matrix, dim=2, weight='weight', g_ca=g_ca, g_cr=g_cr, seed=seed)
    # print(pos)
    # pos1 = nx.spring_layout(Gn_RD)
    # print(pos1)
    plt.figure(figsize=(10, 10))
    plt.grid(True, which='both', axis='both')
    nx.draw(G1, pos=pos, node_size=node_size, with_labels=True, **options)
    for i in range(len(edgedict)):
        if weight[i] > 0:
            nx.draw_networkx_edges(G1, pos=pos, edgelist=edgedict[i + 1], edge_color='forestgreen')
        else:
            nx.draw_networkx_edges(G1, pos=pos, edgelist=edgedict[i + 1], edge_color='red')


# 社交机器人可视化
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import networkx as nx
import os


# 区分边
def distinguish_edges(G_matrix, tag):
    # 人与人之间的边
    edge_human = []
    # 机器人与机器人之间的边
    edge_robot = []
    # 人与机器人之间的边
    edge_dif = []
    for start, line in enumerate(G_matrix):
        for end, value in enumerate(line):
            if value == 1:
                if tag[start] == tag[end] == 0:
                    edge_human.append((start, end))
                elif tag[start] == tag[end] == 1:
                    edge_robot.append((start, end))
                elif tag[start] != tag[end]:
                    edge_dif.append((start, end))
    return edge_human, edge_robot, edge_dif


# 计算度值
def degree(G_matrix):
    degree = G_matrix.sum(axis=1)
    degree_norm = (degree - min(degree)) / (max(degree) - min(degree))  # 归一化
    return degree_norm


# 长方形排坐标
def generate_center_pos(group, group_num):
    center_pos = []
    sizes = []
    for index, i in enumerate(group):
        sizes.append(np.sqrt(len(i) / np.pi))
    w = (int(max(sizes)) + 1)
    h = w
    if group_num % 2 == 0:
        layer = group_num / 2
    else:
        layer = int(group_num / 2) + 1

    margin = 2
    pos_up = []
    tmp = int(group_num / 2) - 1
    for i in range(group_num):
        x = w / 2 + margin * i + w * i
        if i < layer:
            y = h / 2 + margin * i + h * i
        else:
            y = h / 2 + margin * tmp + h * tmp
            tmp -= 1
        pos_up.append([x, y])

    for i in pos_up:
        center_pos.append(i)
        x = i[0]
        y = 0 - i[1] - margin
        center_pos.append([x, y])
    group_radius = w
    return center_pos, group_radius


# 生成群组字典
def generate_group_dict(center_pos, group):
    # define group dict
    # get center points
    group_dict = {}
    num = 0
    for index, list in enumerate(group):
        group_dict[index] = {'points': list}
        pos = center_pos[num]
        num += 1
        group_dict[index]['center_point'] = pos
    return group_dict


# 进行坐标缩放
def rescale_layout(pos, scale=1):
    pos -= pos.mean(axis=0)
    lim = np.abs(pos).max()
    if lim > 0:
        pos *= scale / lim
    return pos


def pos_updata_1(G, pos, k, step_length):  # 根据节点之间的斥力和相邻节点的引力更新节点坐标
    displacement = np.zeros((2, pos.shape[0]))
    for i in range(G.shape[0]):
        # if i not in center_point:
        delta = (pos[i] - pos).T
        distance = np.sqrt((delta ** 2).sum(axis=0))

        distance = np.where(distance < 0.01, 0.01, distance)
        Ai = np.asarray(G[i, :])
        #             print('Ai', Ai)
        displacement[:, i] += \
            (delta * (k * k / distance ** 2 - Ai * distance / k)).sum(axis=1)
        # update positions

    length = np.sqrt((displacement ** 2).sum(axis=0))
    length = np.where(length < 0.01, 0.2, length)
    delta_pos = (displacement * step_length / length).T
    # pos += delta_pos
    pos1 = pos + delta_pos
    return pos1


# 生成坐标
def generate_layout(G, group_dict, t=None, C=1, g_cr=0.4, g_ca=5.0, iterations=600, threshold=1e-5):
    pos = dict()
    for group in group_dict.values():
        G_group = []
        G_group = G[group['points']][..., group['points']]
        pos_group = []
        pos_group = np.asarray(np.random.rand(len(G_group), 2), dtype=float)

        t = max(max(pos_group.T[0]) - min(pos_group.T[0]), max(pos_group.T[1]) - min(pos_group.T[1])) * 0.1
        dt = t / float(iterations + 1)

        k = np.sqrt(50 / len(group['points']))

        for iter in range(iterations):
            pos_1 = pos_updata_1(G_group, pos_group.copy(), k, t)
            err = np.linalg.norm(pos_1 - pos_group) / len(G_group)
            pos_group = pos_1.copy()
            if err < threshold:
                break
            t -= dt

        pos_group = rescale_layout(pos_group, scale=2) + group['center_point']
        pos_ = dict(zip(group['points'], pos_group))
        pos.update(pos_)

    return pos


def Generate_G1(file_path, direction):
    tmp_G = nx.read_edgelist(file_path, nodetype=int)
    if direction == False:
        # 创建新的图将节点排序
        G1 = nx.Graph()
        G1.add_nodes_from(sorted(tmp_G.nodes(data=True)))
        G1.add_edges_from(tmp_G.edges(data=True))
    elif direction == True:
        G1 = nx.DiGraph()
        G1.add_nodes_from(sorted(tmp_G.nodes(data=True)))
        G1.add_edges_from(tmp_G.edges(data=True))
    return G1


# 展示详细图
def Show_all(file_path, G_matrix, tag, pos, group_dict, options, direction, save=False):
    plt.figure(figsize=(8, 8))
    G1 = Generate_G1(file_path, direction=direction)

    # 计算节点度值
    feature = degree(G_matrix)
    sizes = [x * 300 + 40 for x in feature]
    dis_size = []  # 区分节点大小
    for i in group_dict.values():
        tmp_size = []
        # print(i)
        for j in i['points']:
            tmp_size.append(sizes[j])
        dis_size.append(tmp_size)
    nx.draw(G1, pos, with_labels=True, edgelist=[], node_size=sizes, font_size=8, node_color=[])
    for i in range(len(group_dict)):
        color_list = []
        for node in group_dict[i]['points']:
            if tag[node] == 0:
                color_list.append(options['node_colors'][0])
            else:
                color_list.append(options['node_colors'][1])
        nx.draw_networkx_nodes(G1, pos, nodelist=group_dict[i]['points'], node_color=color_list, node_size=dis_size[i])

    if direction == False:
        arrows = False
        arrow_size = 0
    else:
        arrows = True
        arrow_size = 5

    edge_human, edge_robot, edge_dif = distinguish_edges(G_matrix, tag)
    nx.draw_networkx_edges(G1, pos, alpha=0.3, edgelist=edge_human, width=0.5, edge_color='#146bc2', arrows=arrows,
                           arrowsize=arrow_size)
    nx.draw_networkx_edges(G1, pos, alpha=0.3, edgelist=edge_robot, width=0.5, edge_color='#bf1010', arrows=arrows,
                           arrowsize=arrow_size)
    nx.draw_networkx_edges(G1, pos, alpha=0.3, edgelist=edge_dif, width=0.5, edge_color='#6bbe71', arrows=arrows,
                           arrowsize=arrow_size)

    if save == True:
        os_path = os.path.dirname(
            os.path.abspath(__file__))  # os_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 是上级目录
        image_save_path = os_path + '/images'
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        plt.savefig(f'{image_save_path}/all_group_{len(group_dict) / 2}.jpg', dpi=300)
    plt.show()


# 展示缩略图
def Show_mini(file_path, direction, group_dict, group_tag, options, save):
    G1 = Generate_G1(file_path, direction)
    G_matrix = nx.adjacency_matrix(G1).todense()

    # 绘制缩略图
    # 直接用详细图中每个群组中心点的位置
    min_group_pos = [group_dict[i]['center_point'] for i in range(len(group_dict))]

    # 重新定义群组的标签 偶数为人类用户群组，奇数为社交机器人用户群组
    new_group_tag = [0 for i in range(len(group_tag))]
    for i in group_dict.keys():
        for j in group_dict[i]['points']:
            new_group_tag[j] = i

    # 区分连边
    # 统计每个群组之间边的个数，方便设置边的粗细
    min_edge_dict = {}
    for i in range(len(G_matrix)):
        row = G_matrix[i]
        for j in range(len(row)):
            if row[j] == 0:
                continue
            min_edge = (new_group_tag[i], new_group_tag[j])
            if min_edge not in min_edge_dict:
                min_edge_dict[min_edge] = {}
                min_edge_dict[min_edge]['num'] = 0
            min_edge_dict[min_edge]['num'] += 1

    # 设置边的颜色
    for i in min_edge_dict.keys():
        group1 = i[0]
        group2 = i[1]
        if group1 % 2 == 0 and group2 % 2 == 0:
            min_edge_dict[i]['color'] = options['edge_colors'][0]
        elif group1 % 2 != 0 and group2 % 2 != 0:
            min_edge_dict[i]['color'] = options['edge_colors'][1]
        else:
            min_edge_dict[i]['color'] = options['edge_colors'][2]

    if direction == False:
        G1_min = nx.Graph()
        G1_min.add_nodes_from([i for i in range(len(min_group_pos))])
        G1_min.add_edges_from(min_edge_dict.keys())
    else:
        G1_min = nx.DiGraph()
        G1_min.add_nodes_from([i for i in range(len(min_group_pos))])
        G1_min.add_edges_from(min_edge_dict.keys())

    node_colors = []
    for i in range(len(min_group_pos)):
        if i % 2 == 0:
            node_colors.append(options['node_colors'][0])
        else:
            node_colors.append(options['node_colors'][1])

    # 设置节点/边属性
    # 边的粗细
    min_line_weight = [min_edge_dict[i]['num'] * 0.2 + 1 for i in G1_min.edges()]
    min_line_color = [min_edge_dict[i]['color'] for i in G1_min.edges()]
    min_node_sizes = [len(value['points']) * 35 + 50 for value in group_dict.values()]

    plt.figure(figsize=(8, 8))
    nx.draw(G1_min, min_group_pos, with_labels=True, font_size=9, node_color=node_colors, edgelist=[],
            node_size=min_node_sizes)
    # connectionstyle让边弯曲一点
    if direction == True:
        arrows = True
        arrowsize = 20
    else:
        arrows = False
        arrowsize = 0
    nx.draw_networkx_edges(G1_min, min_group_pos, edge_color=min_line_color, width=min_line_weight, arrows=arrows,
                           arrowsize=arrowsize, connectionstyle='arc3,rad=0.1', alpha=0.7)

    # 是否保存
    if save == True:
        os_path = os.path.dirname(
            os.path.abspath(__file__))  # os_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 是上级目录
        image_save_path = os_path + '/images'
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        plt.savefig(f'{image_save_path}/min_group_{len(group_dict) / 2}.jpg', dpi=300)
    plt.show()


def show_robot_group(file_path, G, tag=None, group_tag=None, group=None, direction=False, save=False, show=True,
                     **options):
    '''
    展示社交机器人网络中的分组情况
    还区分组内的人类用户和社交机器人用户

    参数
    -----------
    file_path : 读取的文件路径
    G : read_file()返回的图邻接表
    tag : 节点标签，人0/机器人1
    group_tag : 群组标签（0-n） [0,1,...]
    group : 群组列表，和group_tag有一个就行[[1,2,3], [4,5,6]]
    direction : 方向，True:有向图/False:无向图
    save : 是否保存结果图片，True:保存/False:不保存，保存的结果路径：当前文件的路径下自动创建images文件夹

    options : 节点和边的颜色信息
            节点颜色顺序：人、机器人
            边的颜色顺序：人之间的边， 机器人之间的边，人与机器人之间的边
    '''

    G_matrix = G.toarray()  # adjancy matrix
    N = len(G_matrix)  # 节点个数

    # 设置节点和边的颜色
    if 'node_colors' not in options:
        options['node_colors'] = ['#60afff', '#ff6073']
    if 'edge_colors' not in options:
        options['edge_colors'] = ['#146bc2', '#bf1010', '#6bbe71']

    # 处理group
    if group_tag is None and group is None:
        return print('-ERROR No Group')

    # 处理只给了group标签
    if group_tag is not None and group is None:
        group_num = len(set(group_tag))
        group = []
        for _ in range(group_num):
            group.append([])
        for key, value in enumerate(group_tag):
            # print(f'{key} : {value}')
            group[value].append(key)
    # 处理只给了group分组
    if group is not None and group_tag is None:
        group_tag = [_ for _ in range(N)]
        for key, value in enumerate(group):
            for i in value:
                group_tag[i] = key

    new_group = []
    for list in group:
        h_list = [j for j in list if tag[j] == 0]
        r_list = [j for j in list if tag[j] == 1]
        new_group.append(h_list)
        new_group.append(r_list)

    center_pos, group_radius = generate_center_pos(new_group, len(new_group) // 2)
    group_dict = generate_group_dict(center_pos, new_group)
    # 计算节点坐标
    pos = generate_layout(G_matrix, group_dict, t=None, C=1, g_cr=1.2, g_ca=1.8, iterations=100, threshold=1e-5)

    if show == True:
        # 展示详细图
        Show_all(file_path, G_matrix, tag, pos, group_dict, options, direction, save)
        # 展示缩略图
        Show_mini(file_path, direction, group_dict, group_tag, options, save)

    return pos


def read_file(file_path):
    row = []
    col = []
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                r, c = line.strip().split(' ')
            except:
                r, c = line.strip().split('	')
            row.append(int(r))
            col.append(int(c))
            data.append(1)
    a = max(max(row), max(col)) + 1
    return coo_matrix((data, (row, col)), shape=(a, a))


def Social_Bot_Network_Visualization(file_path, tag_file, group_tag_file,
                                     options):  # 其他参数字典，节点颜色：人、机器人，边颜色：人之间、机器人之间、人与机器人 options = {'node_colors' : ['#60afff', '#ff6073'], 'edge_colors' : ['#146bc2', '#bf1010', '#6bbe71']}

    G = read_file(file_path)
    # 节点标签
    tag = pd.read_csv(tag_file, header=None)
    tag = list(tag[0])
    # 组别标签
    group_tag = pd.read_csv(group_tag_file, header=None)
    group_tag = list(group_tag[0])
    # 其他参数字典，节点颜色：人、机器人，边颜色：人之间、机器人之间、人与机器人

    pos = show_robot_group(file_path, G, tag=tag, group_tag=group_tag, direction=False, save=True, show=True, **options)