
import numpy as np
import pickle as pkl
import networkx as nx
import tensorflow as tf
from scipy import sparse
import scipy.sparse as sp
import scipy.io as scio
from scipy.sparse import identity
from scipy.sparse import coo_matrix, hstack
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import jaccard_similarity_score
import sys
#import pdb
import random
from random import shuffle

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

   # pdb.set_trace()
    #print('====graph analysis======')

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    #pdb.set_trace()
    #print('====input analysis======')
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  
    idx_test = test_idx_range.tolist()
    #idx_train = range(len(y)+1068)
    idx_train = range(len(y)+1068)
    #idx_train = nd0[0].tolist()

    idx_val = range(len(y)+1068, len(y)+1068+500)

    train_mask = sample_mask(idx_train, labels.shape[0]) # the training index is true, others is false
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

"""
# enlarge to second-order neighbors
def reorgonize_features(features,adj):
     
    nd = np.array(adj.sum(1))
    nd_avg = 5
    nd_med = 2
    adj = adj.toarray()
    features = features.toarray()
    adj0 = np.zeros(adj.shape)
    
    # the first one
 
    addr0 = np.where(adj[0]==1)
    if nd[0] > nd_avg:
        aa0 = np.vstack((features[0],features[addr0[0]]))
        bb0 = cosine_similarity(aa0)
        cc0 = bb0[0][1:] 
        cc_add =  np.argsort(cc0)
        addr_0 = addr0[0][cc_add[0:int(len(cc_add)/2)]]
        addr_1 = addr0[0][cc_add[int(len(cc_add)/2):]]
        fea_1 = features[addr_1].sum(0)
        #fea_1 = np.vstack((features[0],features[addr_1])).sum(0)
        fea_0 = features[addr_0].sum(0)
        fea_mer = np.vstack((features[0],fea_1,fea_0))
        fea_mer = np.expand_dims(fea_mer,0)
        new_features = fea_mer
    else:
        dd0 = np.vstack((adj[0],adj[addr0[0]])).sum(0)
        addr_2 = np.where(dd0 > 0 )

        if len(addr_2[0])>nd_med:

            ee0 = np.vstack((features[0], features[addr_2[0]]))
            ff0 = cosine_similarity(ee0)
            gg0 = ff0[0][1:]
            gg_add = np.argsort(gg0)
            addr_3 = addr_2[0][gg_add[0:int(len(gg_add)/2)]]
            addr_4 = addr_2[0][gg_add[int(len(gg_add)/2):]]
            fea_4 = features[addr_4].sum(0)
            #fea_4 = np.vstack((features[0],features[addr_4])).sum(0)
            fea_3 = np.vstack((features[0],features[addr_3])).sum(0)
            fea_mer = np.vstack((features[0],fea_4,fea_3))
            fea_mer = np.expand_dims(fea_mer,0)
            new_features = fea_mer
        
        else:
            gap = nd_med-int(nd[0])+1   # gap+1, consider merge node self features
            fea_mer = features[addr0]
            for k in range(gap):
                fea_mer = np.concatenate([fea_mer,features[0].reshape(1,features.shape[1])])          
            fea_mer = np.expand_dims(fea_mer,0)
            new_features = fea_mer
             
        
    # the latter part
    for i in range(1,len(nd)):
        addr = np.where(adj[i]==1)
        if nd[i]> nd_avg:
            aa = np.vstack((features[i],features[addr[0]]))
            bb = cosine_similarity(aa)
            cc = bb[0][1:] 
            cc_add = np.argsort(cc) # shengxu, and get the address
            addr_0 = addr[0][cc_add[0:int(len(cc_add)/2)]]  #round（）
            addr_1 = addr[0][cc_add[int(len(cc_add)/2):]]
            #fea_1 = np.vstack(( features[i],features[addr_1])).sum(0)
            fea_1 = features[addr_1].sum(0)
            fea_0 = features[addr_0].sum(0)
            fea_mer = np.vstack((features[i],fea_1,fea_0))
            fea_mer = np.expand_dims(fea_mer,0)
            new_features = np.concatenate((new_features,fea_mer))
                
        else:

            dd1 = np.vstack((adj[i],adj[addr[0]])).sum(0)
            addr_2 = np.where(dd1 > 0 )

            if len(addr_2[0])>nd_med:
                ee0 = np.vstack((features[0], features[addr_2[0]]))
                ff0 = cosine_similarity(ee0)
                gg0 = ff0[0][1:]
                gg_add = np.argsort(gg0)
                addr_3 = addr_2[0][gg_add[0:int(len(gg_add)/2)]]
                addr_4 = addr_2[0][gg_add[int(len(gg_add)/2):]]
                fea_4 = features[addr_4].sum(0)
                #fea_4 = np.vstack((features[0],features[addr_4])).sum(0)
                fea_3 = features[addr_3].sum(0)
                fea_mer = np.vstack((features[0],fea_4,fea_3))
                fea_mer = np.expand_dims(fea_mer,0)
                new_features = np.concatenate((new_features,fea_mer))
            
            else:
                #pdb.set_trace()
                #print('====== reorgonize  features ======')
                gap = nd_med-int(nd[i])+1   # gap+1, consider merge node self features
                fea_mer = features[addr]
                for k in range(gap):
                    fea_mer = np.concatenate([fea_mer,features[0].reshape(1,features.shape[1])])          
                fea_mer = np.expand_dims(fea_mer,0)
                new_features = np.concatenate((new_features,fea_mer))

    new_features = np.transpose(new_features, (0, 2, 1))    
    return new_features,adj0     
"""
"""
def reorgonize_features(features,adj):
     
    nd = np.array(adj.sum(1))
    nd_avg = 2
    adj = adj.toarray()
    features = features.toarray()
    adj0 = np.zeros(adj.shape)
    
    # the first one
 
    addr0 = np.where(adj[0]==1)
    if nd[0] > nd_avg:
        aa0 = np.vstack((features[0],features[addr0[0]]))
        bb0 = cosine_similarity(aa0)
        cc0 = bb0[0][1:] 
        cc_add =  np.argsort(cc0)
        addr_0 = addr0[0][cc_add[0:int(len(cc_add)/2)]]
        addr_1 = addr0[0][cc_add[int(len(cc_add)/2):]]
        fea_1 = features[addr_1].sum(0)
        #fea_1 = np.vstack((features[0],features[addr_1])).sum(0)
        fea_0 = features[addr_0].sum(0)
        fea_mer = np.vstack((features[0],fea_1,fea_0))
        fea_mer = np.expand_dims(fea_mer,0)
        new_features = fea_mer
    else:
        gap = nd_avg-int(nd[0])+1  # gap+1, consider merge node self features
        fea_mer = features[addr0]
        for k in range(gap):
            fea_mer = np.concatenate([fea_mer,features[0].reshape(1,features.shape[1])])          
        fea_mer = np.expand_dims(fea_mer,0)
        new_features = fea_mer         
        
    # the latter part
    for i in range(1,len(nd)):
        addr = np.where(adj[i]==1)
        if nd[i]> nd_avg:
            aa = np.vstack((features[i],features[addr[0]]))
            bb = cosine_similarity(aa)
            cc = bb[0][1:] 
            cc_add = np.argsort(cc) # shengxu, and get the address
            addr_0 = addr[0][cc_add[0:int(len(cc_add)/2)]]  #round（）
            addr_1 = addr[0][cc_add[int(len(cc_add)/2):]]
            #fea_1 = np.vstack((features[i],features[addr_1])).sum(0)
            fea_1 = features[addr_1].sum(0)
            fea_0 = features[addr_0].sum(0)
            fea_mer = np.vstack((features[i],fea_1,fea_0))
            fea_mer = np.expand_dims(fea_mer,0)
            new_features = np.concatenate((new_features,fea_mer))
                
        else:
            gap = nd_avg-int(nd[i])+1   # gap+1, consider merge node self features
            fea_mer = features[addr]
            for k in range(gap):
                fea_mer = np.concatenate([fea_mer,features[0].reshape(1,features.shape[1])])          
            fea_mer = np.expand_dims(fea_mer,0)
            new_features = np.concatenate((new_features,fea_mer))
            
    new_features = np.transpose(new_features, (0, 2, 1))    
    return new_features,adj0     
"""



def reorgonize_features(features,adj):
    
    nd = np.array(adj.sum(1))
    nd_avg = 1
    print('==== the number of the neighbors ====', nd_avg)
    adj = adj.toarray()
    features = features.toarray()
    adj0 = np.zeros(adj.shape)

    ### the first one   
    nd0 = nd[0]
    addr = np.where(adj[0]>0)[0]
    aa = addr
    if int(nd0)> nd_avg:
 
        # ***** choose by calculate the common neighbors *****
        # for k in range(int(nd0)):
        #     addr = np.where(adj[0]>0)[0]
        #     addr00 = np.where(adj[addr[k]]>0)
        #     inter0 = np.intersect1d(addr,addr00)
        #     if len(inter0) > 0:
        #         aa[k] = aa[k]
        #     else:
        #         aa[k] = 0
        
        
        # if len(np.where(aa>0)[0]) >= nd_avg:
        #     addr_2 = aa[np.where(aa>0)[0]]           
        #     addr_1 = np.random.choice(addr_2,int(nd_avg),replace = False)
        #     fea_mer0 = np.vstack((features[0],features[addr_1]))
        #     fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
        #     new_features = fea_mer0 
        
        # else:
        #     addr_2 = np.where(aa>0)
        #     if len(addr_2[0]) > 0:
        #         fea_mer00 = np.vstack((features[0],features[aa[addr_2]]))
        #         gap = nd_avg - len(addr_2[0])
        #         addr_3 = np.random.choice(addr[0],int(gap),replace = False)
        #         fea_mer0 = np.vstack((fea_mer00,features[addr_3]))
        #         fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
        #         new_features = fea_mer0 

        #     else:
        #         addr1 = np.random.choice(addr[0],int(nd_avg),replace = False) #choose the real address directly
        #         fea_mer0 = np.empty([nd_avg,features.shape[1]])
        #         fea_mer0 = np.vstack((features[0],features[addr1]))  #(5, 1433)
        #         fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
        #         new_features = fea_mer0   
        #         fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
        #         new_features = fea_mer0  
        # # adj0[0][addr1] = 1        
        # ***** choose by calculate the common neighbors end *****

        # ***** choose 2 neighbors(similarity), add left neighbors ****
        # aa = features[addr[0]]
        # aa0 = sp.vstack((features[0],aa))
        # bb = cosine_similarity(aa0)
        # cc = bb[0][1:]        
        # cc_addr1 = np.argpartition(cc,-int(nd_avg))[-int(nd_avg):] # after get the most relevant node and get the real address
        # addr1 = addr[0][cc_addr1]        
        # fea_left0 = features[addr].sum(0) - features[addr1].sum(0)
        # fea_mer0 = np.vstack((features[0],features[addr1],fea_left0))
        # fea_mer0 = np.expand_dims(fea_mer0,0)
        # new_features = fea_mer0
        # ***** choose 2 neighbors, add left neighbors and merge****
       
        # ***** choose 2 neighbors(randomly), add left neighbors ****
        # addr1 = np.random.choice(addr[0],int(nd_avg),replace = False) #choose the real address directly
        # fea_left0 = features[addr].sum(0) - features[addr1].sum(0)
        # fea_mer0 = np.vstack((features[0],features[addr1]))
        # fea_mer0 = np.expand_dims(fea_mer0,0)
        # new_features = fea_mer0
        # ***** choose 2 neighbors, add left neighbors and merge****
                
        # ***** choose by the features similarity *******      
        # aa = features[addr[0]]
        # aa0 = np.vstack((features[0],aa))
        # bb = cosine_similarity(aa0)
        # cc = bb[0][1:]        
        # #cc_addr1 = np.argpartition(cc,-int(nd_avg))[-int(nd_avg):] # get the top nd_avg neighbor
        #                                                            # from lower correlation to higher correaltion,
        #                                                            #after get the most relevant node and get the real address
        # cc_addr1 = np.argpartition(cc,-int(nd_avg))[0:nd_avg] # get lower correlation neighbors
        
        # addr1 = addr[0][cc_addr1]
        # fea_mer0 = np.empty([nd_avg,features.shape[1]])
        # fea_mer0 = np.vstack((features[addr1],features[0]))   
        # fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
        # new_features = fea_mer0
        # # adj0[0][addr1] = 1      
        # ***** choose by the features similarity end *******       
        
        # ***** choose randomly *****
        print('==== choose neighbor randomly ===')
        #addr1 = np.random.choice(addr[0],int(nd_avg),replace = False) #choose the real address directly
        addr1 = np.random.choice(addr[0],int(nd_avg)) #choose the real address totally random
        fea_mer0 = np.empty([nd_avg,features.shape[1]])
        fea_mer0 = np.vstack((features[0],features[addr1]))  #(5, 1433)
        fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
        new_features = fea_mer0    
        # adj0[0][addr1] = 1        
        # ***** choose randomly end *****
 
        # ***** add all neibhors and learning to merge****
        # fea_left0 = features[addr].sum(0)
        # fea_mer0 = np.vstack((fea_left0,features[0]))
        # fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
        # new_features = fea_mer0
        # ***** add all the left neibhors and learning to merge****

    else:
        gap = nd_avg-int(nd[0])+1   # gap+1, consider merge node self features
                                    # gap+2, consider merge node self and the left added neighbors
        fea_mer1 = features[addr]
        for k in range(gap):
            fea_mer1 = np.concatenate([features[0].reshape(1,features.shape[1]),fea_mer1])
        
        fea_mer1 = np.expand_dims(fea_mer1,0)
        new_features = fea_mer1
             

    ### the later part

    for i in range(1,len(nd)):
        addr0 = np.where(adj[i] == 1) # find all one
        aa = addr0[0]
        if int(nd[i]) >= nd_avg:
            
        # ***** choose by calculate the common neighbors *****
            # for k in range(int(nd[i])):               
            #     addr = np.where(adj[i]>0)[0]
            #     addr00 = np.where(adj[addr[k]]>0)
            #     inter0 = np.intersect1d(addr,addr00)
            #     if len(inter0) > 0:
            #         aa[k] = aa[k]
            #     else:
            #         aa[k] = 0

            # if len(np.where(aa>0)[0]) >= nd_avg:
                
            #     addr_2 = aa[np.where(aa>0)[0]]           
            #     addr_1 = np.random.choice(addr_2,int(nd_avg),replace = False)
            #     fea_mer0 = np.vstack((features[0],features[addr_1]))
            #     fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
            #     new_features = np.concatenate((new_features,fea_mer0))         
            # else:
            #     addr_2 = np.where(aa>0)

            #     if len(addr_2[0]) > 0:
            #         fea_mer00 = np.vstack((features[i],features[aa[addr_2]]))
            #         gap = nd_avg - len(addr_2[0])
            #         addr_3 = np.random.choice(addr,int(gap),replace = False)
            #         fea_mer0 = np.vstack((fea_mer00,features[addr_3]))
            #         fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
            #         new_features = np.concatenate((new_features,fea_mer0)) 
            #     else:
            #         addr1 = np.random.choice(addr,int(nd_avg),replace = False) #choose the real address directly
            #         # fea_mer0 = np.empty([nd_avg,features.shape[1]])
            #         fea_mer0 = np.vstack((features[i],features[addr1]))  #(5, 1433)
            #         # fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
            #         # new_features = fea_mer0   
            #         fea_mer0 = np.expand_dims(fea_mer0,0) #(1, 5, 1433)
            #         new_features = np.concatenate((new_features,fea_mer0))
                     
            # # adj0[0][addr1] = 1        
            # ***** choose by calculate the common neighbors end *****

        # ***** choose 2 neighbors(similarity), add left neighbors**** 
            # aa = features[addr0[0]]
            # aa0 = sp.vstack((features[i],aa))
            # bb = cosine_similarity(aa0)
            # cc = bb[0][1:]        
            # cc_addr1 = np.argpartition(cc,-int(nd_avg))[-int(nd_avg):]
            # addr2 = addr0[0][cc_addr1]       
            # fea_left1 = features[addr0].sum(0) - features[addr2].sum(0)
            # fea_mer2 = np.vstack((features[i],features[addr2],fea_left1))
            # fea_mer2 = np.expand_dims(fea_mer2,0)
            # new_features = np.concatenate((new_features,fea_mer2))
        # ***** choose 2 neighbors, add left neighbors and merge****
            
        # ***** choose 2 neighbors(randomly), add left neighbors****            
            # addr2 = np.random.choice(addr0[0],int(nd_avg),replace = False) #choose the real address directly
            # fea_left1 = features[addr0].sum(0) - features[addr2].sum(0)
            # fea_mer2 = np.vstack((features[i],features[addr2]))
            # fea_mer2 = np.expand_dims(fea_mer2,0)
            # new_features = np.concatenate((new_features,fea_mer2))
        # ***** choose 2 neighbors, add left neighbors and merge****
            

        # ***** choose by the features similarity *******            
            # aa = features[addr0[0]]
            # aa0 = np.vstack((features[i],aa))
            # bb = cosine_similarity(aa0)
            # cc = bb[0][1:]        
            # #cc_addr1 = np.argpartition(cc,-int(nd_avg))[-int(nd_avg):]
            # cc_addr1 = np.argpartition(cc,-int(nd_avg))[0:nd_avg]
            # addr1 = addr0[0][cc_addr1]
            # fea_mer3 = np.empty([nd_avg,features.shape[1]])
            # fea_mer3 = np.vstack((features[i],features[addr1]))  #(5, 1433)
            # fea_mer3 = np.expand_dims(fea_mer3,0) #(1, 5, 1433)
            # new_features = np.concatenate((new_features,fea_mer3))
            # # adj0[i][addr1] = 1                                             
        # ***** end choose by the features similarity *******   
            

        # **** choose randomly ******
            addr1 = np.random.choice(addr0[0],int(nd_avg),replace = False) 
            fea_mer3 = np.empty([nd_avg,features.shape[1]])
            fea_mer3 = np.vstack((features[i],features[addr1]))
            fea_mer3 = np.expand_dims(fea_mer3, 0)
            new_features = np.concatenate((new_features,fea_mer3))
            # adj0[i][addr1] = 1
        # **** choose randomly ******

        # ***** add all the left neibhors and learning to merge****
            # fea_left1 = features[addr0].sum(0)
            # fea_mer1 = np.vstack((fea_left1,features[i]))
            # fea_mer1 = np.expand_dims(fea_mer1,0) #(1, 5, 1433)
            # new_features = np.concatenate((new_features,fea_mer1))             
        # ***** add all the left neibhors and learning to merge****

        else:
            gap = nd_avg-int(nd[i])+1
            fea_mer4 = features[addr0] 
            for k in range(gap):  
                fea_mer4 = np.concatenate([features[i].reshape(1,features.shape[1]),fea_mer4],0)                             
            fea_mer4 = np.expand_dims(fea_mer4, 0)
            new_features = np.concatenate((new_features,fea_mer4))             
    
    new_features = np.transpose(new_features, (0, 2, 1)) 
    #pdb.set_trace()   
    return new_features, adj0


def preprocess_features(features):

    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features) ## type(features) scipy.sparse.csr.csr_matrix'
    #return sparse_to_tuple(features)
    return features
    """*************************analysis_node_degree version 1*************************"""



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
##  add the node self feature and normalize the primary adjcency matrix


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    
    adj_normalized0 = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized0)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})

    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
