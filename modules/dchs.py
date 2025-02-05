import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import geotorch

        
class NirvanaOpenset_loss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        num_subcenters (int): number of subcenters per class
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=4, feat_dim=128, precalc_centers=None, margin=48.0, Expand=200):
        super(NirvanaOpenset_loss, self).__init__()
        self.num_classes = num_classes
        self.num_centers = self.num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.E = num_classes
        if(precalc_centers):
            precalculated_centers = FindCenters(self.feat_dim, self.E)
            precalculated_centers = precalculated_centers[:self.num_classes,:]

        with torch.no_grad():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim, requires_grad=False))
            if(precalc_centers):
                self.centers.copy_(torch.from_numpy(precalculated_centers))
                print('Centers loaded.')

        
    def forward(self, labels, x, x_out, ramp=False):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
            # Fix: label range check
        if labels.max() >= self.num_classes or labels.min() < 0: #Ensures that provided labels are within [0, num_classes-1], since the centers are indexed by class.
            raise ValueError("Labels out of valid range for dchs_loss.")

        batch_size = x.size(0) #retrieve the number of samples in the batch
        
        """– Initializes a distance matrix between features x and each center.
            – The term torch.pow(x, 2).sum(dim=1) is ‖fᵢ‖².
            – The term torch.pow(self.centers, 2).sum(dim=1) is ‖sⱼ‖².
            – They are broadcasted to form part of ‖fᵢ - sⱼ‖² = ‖fᵢ‖² + ‖sⱼ‖² - 2 fᵢ⋅sⱼ."""
        inlier_distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        """Subtracts 2 fᵢ⋅sⱼ from the distance matrix, completing the computation of ‖fᵢ - sⱼ‖²."""
        inlier_distmat.addmm_(x, self.centers.t(),beta=1,alpha=-2)

        """– Picks out the correct distance to the ground-truth center sᵧᵢ for each sample fᵢ.
            – These distances correspond to ‖fᵢ - sᵧᵢ‖² from the paper."""
        intraclass_distances = inlier_distmat.gather(dim=1,index=labels.unsqueeze(1)).squeeze()

        """– Computes the average of these intraclass distances across the batch, matching the first term (‖fᵢ - sᵧᵢ‖²) in the formula."""
        intraclass_loss = intraclass_distances.sum()/(batch_size*self.feat_dim*2.0)

        """– Computes a matrix of differences: (‖fᵢ - sᵧᵢ‖²) - (‖fᵢ - sₖ‖²) for all k ≠ yᵢ."""
        centers_dist_inter = (intraclass_distances.repeat(self.num_centers,1).t() - inlier_distmat)

        """– Creates a one-hot indicator of the ground-truth class for each sample, then inverts it so only non-ground-truth classes are “True.”
            – Ensures we only compute losses for inter-class pairs."""
        mask = torch.logical_not(torch.nn.functional.one_hot(labels.long(),num_classes=self.num_classes))

        """– Applies a margin-based hinge on each difference: max(0, margin + (‖fᵢ - sᵧᵢ‖² - ‖fᵢ - sⱼ‖²)).
            – Multiplies by mask so we only consider j ≠ yᵢ.
            – Divides by a normalization factor (self.num_centersbatch_size2.0) and sums to produce the total interclass triplet loss."""
        interclass_loss_triplet = (1/(self.num_centers*batch_size*2.0))*((self.margin+centers_dist_inter).clamp(min=0)*mask).sum()
        
        if x_out!=None: #checks if there is an outlier data 
            batch_size_out = x_out.size(0) #retrieves number of outlier samples

            #Computes the (‖x_out - sⱼ‖²) distance matrix for all outlier samples and each center (similar to inlier_distmat).
            outlier_distmat = torch.pow(x_out, 2).sum(dim=1, keepdim=True).expand(batch_size_out, self.num_classes) + \
                    torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size_out).t()
            #Completes ‖x_out - sⱼ‖² by subtracting 2·(x_out·sⱼ).
            outlier_distmat.addmm_(x_out, self.centers.t(),beta=1,alpha=-2)
            #Selects the columns corresponding to each sample’s ground-truth class label, aligning outlier distance with the intraclass distance for that label.
            outlier_corresponding_multi_distances = outlier_distmat.index_select(1,labels.long())
      
            if ramp:#Optionally clamps the hinge term between 0 and 60 to stabilize training
                hinge_part = (self.margin+(intraclass_distances-outlier_corresponding_multi_distances)).clamp(min=0.).clamp(max=60.)

                #– Computes the margin-based penalty for separating inlier features from outlier features, normalized by (batch_size × batch_size_out × 2).
                outlier_triplet_multi_loss = (1/(batch_size*batch_size_out*2.0))*hinge_part.sum()
            else:
                ## WORKING PART DO NOT REMOVE..
                outlier_triplet_multi_loss = (1/(batch_size*batch_size_out*2.0))*((self.margin+(intraclass_distances-outlier_corresponding_multi_distances)).clamp(min=0)).sum()
    
            return intraclass_loss, interclass_loss_triplet, outlier_triplet_multi_loss #multiply interclass and outlier by lambda bw 0.05 and 0.95
            # return intraclass_loss, interclass_loss_triplet, outlier_triplet_loss
        else:
            return intraclass_loss, interclass_loss_triplet, None


"""to add lambda for regularization later on:
if x_out is not None:
        ...
        outlier_triplet_multi_loss = ...
        # Combine with regularization
        return intraclass_loss + lambda_reg*interclass_loss_triplet + lambda_reg*outlier_triplet_multi_loss
    else:
        # Combine with regularization
        return intraclass_loss + lambda_reg*interclass_loss_triplet

"""
def FindCenters(k, E=1):
    """
    Calculates "k+1" equidistant points in R^{k}.
    Args:
        k (int) dimension of the space
        E (float) expand factor 
    Returns: 
        Centers (np.array) equidistant positions in R^{k}, shape (k+1 x k)
    """
    
    Centers = np.empty((k+1, k), dtype=np.float32)
    CC = np.empty((k,k), dtype=np.float32)
    Unit_Vector = np.identity(k)
    c = -((1+np.sqrt(k+1))/np.power(k, 3/2))
    CC.fill(c)
    d = np.sqrt((k+1)/k)
    DU = d*Unit_Vector 
    Centers[0,:].fill(1/np.sqrt(k))
    Centers[1:,:] = CC + DU
    
    # Calculate and Check Distances
    # Distances = np.empty((k+1,k), dtype=np.float32)
    # for k, rows in enumerate(Centers):
    #     Distances[k,:] = np.linalg.norm(rows - np.delete(Centers, k, axis=0), axis=1)
    # # print("Distances:",Distances)    
    # assert np.allclose(np.random.choice(Distances.flatten(), size=1), Distances, rtol=1e-05, atol=1e-08, equal_nan=False), "Distances are not equal" 
    return Centers*E

        

def get_l2_pred(features,centers, return_logits=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        if return_logits:
            logits = 1/(1+distmat)
            return pred, logits
        else:
            return pred

def get_l2_pred_b9(features,centers, return_logits=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        if return_logits:
            logits_b9 = 1/(1+F.normalize(distmat,p=2))
            logits = 1/(1+distmat)
            return pred, logits, logits_b9
        else:
            return pred

def accuracy_l2(features,centers,targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = targets.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
                  torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        pred = distmat.argmin(1)
        correct = pred.eq(targets)

        correct_k = correct.flatten().sum(dtype=torch.float32)
        return correct_k * (100.0 / batch_size)  


def accuracy_l2_nosubcenter(features,centers,targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = targets.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        # distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
        #           torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        # distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        distmat = torch.cdist(features, serialized_centers, p=2)
        pred = distmat.argmin(1)
        correct = pred.eq(targets)
        correct_k = correct.flatten().sum(dtype=torch.float32)
        return correct_k * (100.0 / batch_size)    
    
def get_l2_pred_nosubcenter(features,centers, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)

        # distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
        #           torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        # distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        distmat = torch.cdist(features, serialized_centers, p=2)
        pred = distmat.argmin(1)

        return pred

def cosine_similarity(features, centers, target):
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)
        
        pred = torch.empty(batch_size, device=features.device)
        for i in range(batch_size):
            pred[i] = nn.functional.cosine_similarity(features[i].reshape(1,-1), serialized_centers).argmax()
    return pred

def euc_cos(features,centers, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # features are expected in (batch_size,feat_dim)
        # centers are expected in shape (num_classes,num_subcenters,feat_dim)
        batch_size = features.size(0)
        num_classes, feat_dim = centers.shape
        num_centers = num_classes
        
        serialized_centers = centers.view(-1,feat_dim)
        assert num_centers == serialized_centers.size(0)
    
        # distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_centers) + \
        #           torch.pow(serialized_centers, 2).sum(dim=1, keepdim=True).expand(num_centers, batch_size).t()
        # distmat.addmm_(features, serialized_centers.t(),beta=1,alpha=-2)
        # distmat in shape (batch_size,num_centers)
        disteuc = torch.cdist(features, serialized_centers, p=2)
        # pred = distmat.argmin(1)   
        distcos = torch.empty(batch_size, num_classes, device=features.device)
        for i in range(batch_size):
            distcos[i] = nn.functional.cosine_similarity(features[i].reshape(1,-1), serialized_centers)
    return ((1/(2+distcos))*disteuc).argmin(1)


