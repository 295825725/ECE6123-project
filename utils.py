import numpy as np
from scipy.io import loadmat, savemat
from array import array
from skimage import transform as trans
import torch
import torch.nn.functional as F

# ############################## BFM MODEL PREPROCESSING ##########################
# load expression basis
def LoadExpBasis():
	n_vertex = 53215
	Expbin = open('BFM/Exp_Pca.bin','rb')
	exp_dim = array('i')
	exp_dim.fromfile(Expbin,1)
	expMU = array('f')
	expPC = array('f')
	expMU.fromfile(Expbin,3*n_vertex)
	expPC.fromfile(Expbin,3*exp_dim[0]*n_vertex)

	expPC = np.array(expPC)
	expPC = np.reshape(expPC,[exp_dim[0],-1])
	expPC = np.transpose(expPC)

	expEV = np.loadtxt('BFM/std_exp.txt')

	return expPC,expEV

# transfer original BFM09 to our face model
def transferBFM09():
	original_BFM = loadmat('BFM/01_MorphableModel.mat')
	shapePC = original_BFM['shapePC'] # shape basis
	shapeEV = original_BFM['shapeEV'] # corresponding eigen value
	shapeMU = original_BFM['shapeMU'] # mean face
	texPC = original_BFM['texPC'] # texture basis
	texEV = original_BFM['texEV'] # eigen value
	texMU = original_BFM['texMU'] # mean texture

	expPC,expEV = LoadExpBasis()

	# transfer BFM09 to our face model

	idBase = shapePC*np.reshape(shapeEV,[-1,199])
	idBase = idBase/1e5 # unify the scale to decimeter
	idBase = idBase[:,:80] # use only first 80 basis

	exBase = expPC*np.reshape(expEV,[-1,79])
	exBase = exBase/1e5 # unify the scale to decimeter
	exBase = exBase[:,:64] # use only first 64 basis

	texBase = texPC*np.reshape(texEV,[-1,199])
	texBase = texBase[:,:80] # use only first 80 basis

	# our face model is cropped align face landmarks which contains only 35709 vertex.
	# original BFM09 contains 53490 vertex, and expression basis provided by JuYong contains 53215 vertex.
	# thus we select corresponding vertex to get our face model.

	index_exp = loadmat('BFM/BFM_front_idx.mat')
	index_exp = index_exp['idx'].astype(np.int32) - 1 #starts from 0 (to 53215)

	index_shape = loadmat('BFM/BFM_exp_idx.mat')
	index_shape = index_shape['trimIndex'].astype(np.int32) - 1 #starts from 0 (to 53490)
	index_shape = index_shape[index_exp]


	idBase = np.reshape(idBase,[-1,3,80])
	idBase = idBase[index_shape,:,:]
	idBase = np.reshape(idBase,[-1,80])

	texBase = np.reshape(texBase,[-1,3,80])
	texBase = texBase[index_shape,:,:]
	texBase = np.reshape(texBase,[-1,80])

	exBase = np.reshape(exBase,[-1,3,64])
	exBase = exBase[index_exp,:,:]
	exBase = np.reshape(exBase,[-1,64])

	meanshape = np.reshape(shapeMU,[-1,3])/1e5
	meanshape = meanshape[index_shape,:]
	meanshape = np.reshape(meanshape,[1,-1])

	meantex = np.reshape(texMU,[-1,3])
	meantex = meantex[index_shape,:]
	meantex = np.reshape(meantex,[1,-1])

	# other info contains triangles, region used for computing photometric loss,
	# region used for skin texture regularization, and 68 landmarks index etc.
	other_info = loadmat('BFM/facemodel_info.mat')
	frontmask2_idx = other_info['frontmask2_idx']
	skinmask = other_info['skinmask']
	keypoints = other_info['keypoints']
	point_buf = other_info['point_buf']
	tri = other_info['tri']
	tri_mask2 = other_info['tri_mask2']

	# save our face model
	savemat('BFM/BFM_model_front.mat',{'meanshape':meanshape,'meantex':meantex,'idBase':idBase,'exBase':exBase,'texBase':texBase,'tri':tri,'point_buf':point_buf,'tri_mask2':tri_mask2\
		,'keypoints':keypoints,'frontmask2_idx':frontmask2_idx,'skinmask':skinmask})

# utils for face reconstruction
def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# utils for face recognition model
def estimate_norm(lm_68p, H):
    # from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
    """
    Return:
        trans_m            --numpy.array  (2, 3)
    Parameters:
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        H                  --int/float , image height
    """
    lm = extract_5p(lm_68p)
    lm[:, -1] = H - 1 - lm[:, -1]
    tform = trans.SimilarityTransform()
    src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
    tform.estimate(lm, src)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)

    return M[0:2, :]

def estimate_norm_torch(lm_68p, H):
    lm_68p_ = lm_68p.detach().cpu().numpy()
    M = []
    for i in range(lm_68p_.shape[0]):
        M.append(estimate_norm(lm_68p_[i], H))
    M = torch.tensor(np.array(M), dtype=torch.float32).to(lm_68p.device)
    return M

def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    size = torch.Size((image.shape[0], image.shape[1], dsize, dsize))
    grid = F.affine_grid(M, size,align_corners=True)
    output = F.grid_sample(image, grid,align_corners=True)
    return output