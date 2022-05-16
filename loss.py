import soft_renderer as sr
from torch import nn
import torch
import torch.nn.functional as f
import math
from utils import estimate_norm_torch,resize_n_crop
from BFM import BFM_torch
from facenet_pytorch import InceptionResnetV1

class BFMFaceLoss(nn.Module):
    """Decode from the learned parameters to the 3D face model"""
    def __init__(self, parser):
        self.opt=parser
        super(BFMFaceLoss, self).__init__()
        self.device=parser.device
        self.BFM_model = BFM_torch(parser).to(self.device)
        self.renderer = parser.renderer
        self.mse_criterion = nn.MSELoss()
        self.sl1_criterion = nn.SmoothL1Loss()
        self.w_feat = parser.w_feat
        self.w_color = parser.w_color
        self.w_reg = parser.w_reg
        self.w_gamma = parser.w_gamma
        self.w_lm = parser.w_lm
        self.w_reflc = parser.w_reflc
        self.a0 = torch.tensor(math.pi).to(self.device)
        self.a1 = torch.tensor(2*math.pi/math.sqrt(3.0)).to(self.device)
        self.a2 = torch.tensor(2*math.pi/math.sqrt(8.0)).to(self.device)
        self.c0 = torch.tensor(1/math.sqrt(4*math.pi)).to(self.device)
        self.c1 = torch.tensor(math.sqrt(3.0)/math.sqrt(4*math.pi)).to(self.device)
        self.c2 = torch.tensor(3*math.sqrt(5.0)/math.sqrt(12*math.pi)).to(self.device)

        self.reverse_z = torch.eye(3).to(self.device)[None,:,:]
        self.face_net = InceptionResnetV1(pretrained='vggface2').eval()
        for param in self.face_net.parameters():
            param.requires_grad=False
        self.face_net.to(self.device) 
        if parser.feat_method=='method2':
            self.preprocess = lambda x: 2 * x - 1
            self.input_size=112
            

    def split(self, params):
        id_coef = params[:,:80]
        ex_coef = params[:,80:144]
        tex_coef = params[:,144:224]
        angles = params[:,224:227]
        gamma  = params[:,227:254]
        translation = params[:,254:257]
        scale = params[:,257:]
        return id_coef, ex_coef, tex_coef, angles, gamma, translation, scale

    def compute_norm(self, vertices):
        """
        Compute the norm of the vertices
        Input:
            vertices[bs, 35709, 3]
        """
        bs = vertices.shape[0]
        face_id = torch.flip(self.BFM_model.tri.reshape(-1,3)-1, dims=[1])
        point_id = self.BFM_model.point_buf-1
        # [bs, 70789, 3]
        face_id = face_id[None,:,:].expand(bs,-1,-1)
        # [bs, 35709, 8]
        point_id = point_id[None,:,:].expand(bs,-1,-1)
        # [bs, 70789, 3] Gather the vertex location
        v1 = torch.gather(vertices, dim=1,index=face_id[:,:,:1].expand(-1,-1,3).long())
        v2 = torch.gather(vertices, dim=1,index=face_id[:,:,1:2].expand(-1,-1,3).long())
        v3 = torch.gather(vertices, dim=1,index=face_id[:,:,2:].expand(-1,-1,3).long())
        # Compute the edge
        e1 = v1-v2
        e2 = v2-v3
        # Normal [bs, 70789, 3]
        norm = torch.cross(e1, e2)
        # Normal appended with zero vector [bs, 70790, 3]
        norm = torch.cat([norm, torch.zeros(bs, 1, 3).to(self.device)], dim=1)
        # [bs, 35709*8, 3]
        point_id = point_id.reshape(bs,-1)[:,:,None].expand(-1,-1,3)
        # [bs, 35709*8, 3]
        v_norm = torch.gather(norm, dim=1, index=point_id.long())
        v_norm = v_norm.reshape(bs, 35709, 8, 3)
        # [bs, 35709, 3]
        v_norm = f.normalize(torch.sum(v_norm, dim=2), dim=-1)
        return v_norm


    def lighting(self, norm, albedo, gamma):
        """
        Add lighting to the albedo surface
        gamma: [bs, 27]
        norm: [bs, num_vertex, 3]
        albedo: [bs, num_vertex, 3]
        """
        assert norm.shape[0] == albedo.shape[0]
        assert norm.shape[0] == gamma.shape[0]
        bs = gamma.shape[0]
        num_vertex = norm.shape[1]

        init_light = torch.zeros(9).to(self.device)
        init_light[0] = 0.8
        gamma = gamma.reshape(bs,3,9)+init_light

        Y0 = self.a0*self.c0*torch.ones(bs, num_vertex, 1, device=self.device)
        Y1 = -self.a1*self.c1*norm[:,:,1:2]
        Y2 = self.a1*self.c1*norm[:,:,2:3]
        Y3 = -self.a1*self.c1*norm[:,:,0:1]
        Y4 = self.a2*self.c2*norm[:,:,0:1]*norm[:,:,1:2]
        Y5 = -self.a2*self.c2*norm[:,:,1:2]*norm[:,:,2:3]
        Y6 = self.a2*self.c2*0.5/math.sqrt(3.0)*(3*norm[:,:,2:3]**2-1)
        Y7 = -self.a2*self.c2*norm[:,:,0:1]*norm[:,:,2:3]
        Y8 = self.a2*self.c2*0.5*(norm[:,:,0:1]**2-norm[:,:,1:2]**2)
        # [bs, num_vertice, 9]
        Y = torch.cat([Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8],dim=2)

        light_color = torch.bmm(Y, gamma.permute(0,2,1))
        vertex_color = light_color*albedo
        return vertex_color

    def reconst_img(self, params, return_type=None):
        bs = params.shape[0]
        id_coef, ex_coef, tex_coef, angles, gamma, tranlation, scale = self.split(params)

        face_shape = self.BFM_model.get_shape(id_coef, ex_coef)
        face_albedo = self.BFM_model.get_texture(tex_coef) 
        face_shape[:,:,-1] *= -1
        # Recenter the face mesh
        face_albedo = face_albedo.reshape(bs, -1, 3)/255.

        # face model scaling, rotation and translation
        rotation_matrix = self.BFM_model.compute_rotation_matrix(angles)
        face_shape = torch.bmm(face_shape, rotation_matrix)
        # Compute the normal
        normal = self.compute_norm(face_shape)
        
        face_shape = (1+scale[:,:,None])*face_shape
        face_shape = face_shape+tranlation[:,None,:]

        face_albedo = self.lighting(normal, face_albedo, gamma)

        tri = torch.flip(self.BFM_model.tri.reshape(-1,3)-1, dims=[-1])
        face_triangles = tri[None,:,:].expand(bs,-1,-1)

        recon_mesh= sr.Mesh(face_shape,
                            face_triangles,
                            textures=face_albedo,
                            texture_type="vertex")
        recon_img = self.renderer(recon_mesh)
        if return_type == 'all':
            return recon_img, face_shape, face_triangles, face_albedo
        else:
            return recon_img

    def forward(self, params, gt_img, gt_lmk):
        bs = params.shape[0]
        id_coef, ex_coef, tex_coef, angles, gamma, tranlation, scale = self.split(params)

        face_shape = self.BFM_model.get_shape(id_coef, ex_coef)
        face_albedo = self.BFM_model.get_texture(tex_coef) 
        face_shape[:,:,-1] *= -1
        # Recenter the face mesh
        face_albedo = face_albedo.reshape(bs, -1, 3)/255.

        # face model scaling, rotation and translation
        rotation_matrix = self.BFM_model.compute_rotation_matrix(angles)
        face_shape = torch.bmm(face_shape, rotation_matrix)
        # Compute the normal
        normal = self.compute_norm(face_shape)
        
        face_shape = (1+scale[:,:,None])*face_shape
        face_shape = face_shape+tranlation[:,None,:]

        face_albedo = self.lighting(normal, face_albedo, gamma)

        tri = torch.flip(self.BFM_model.tri.reshape(-1,3)-1, dims=[-1])
        face_triangles = tri[None,:,:].expand(bs,-1,-1)

        recon_mesh= sr.Mesh(face_shape,
                            face_triangles,
                            textures=face_albedo,
                            texture_type="vertex")
        recon_img = self.renderer(recon_mesh)
        recon_lmk = recon_mesh.vertices[:, self.BFM_model.keypoints.long(), :]

        # Compute loss
        # remove the alpha channel
        mask = (recon_img[:,-1:,:,:].detach() > 0).float()
        
        # Image loss
        img_loss = self.mse_criterion(recon_img[:,:3,:,:], gt_img*mask)
        # image loss 2
        # img_loss = torch.sqrt(1e-6 + torch.sum((recon_img[:,:3,:,:] - gt_img) ** 2, dim=1, keepdims=True)) * mask
        # img_loss = torch.sum(img_loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))

        # Landmark loss
        recon_lmk_2D_rev = (recon_lmk[:,:,:2]+1)*250./2.
        recon_lmk_2D = (recon_lmk[:,:,:2]+1)*250./2.
        recon_lmk_2D[:,:,1] = 250.-recon_lmk_2D_rev[:,:,1] 
        lmk_loss = self.sl1_criterion(recon_lmk_2D, gt_lmk.float())

        recon_feature = self.face_net(recon_img[:,:3,:,:])
        gt_feature = self.face_net(gt_img*mask)
        # perceptual loss 1
        if self.opt.feat_method=='method1':
            feat_loss = self.mse_criterion(recon_feature, gt_feature)
        else:            
        # perceptual loss 2
            # trans_m = estimate_norm_torch(recon_lmk_2D, gt_img.shape[-2])  
            # image = self.preprocess(resize_n_crop(recon_img[:,:3,:,:], trans_m, self.input_size))
            # recon_feature = F.normalize(self.face_net(image), dim=-1, p=2)            
            # trans_M = estimate_norm_torch(gt_lmk, gt_img.shape[-2])
            # image = self.preprocess(resize_n_crop(gt_img, trans_M, self.input_size))
            # gt_feature = F.normalize(self.face_net(image), dim=-1, p=2)
            
            cosine_d = torch.sum(recon_feature * gt_feature, dim=-1)
            # assert torch.sum((cosine_d > 1).float()) == 0
            feat_loss =  torch.sum(1 - cosine_d) / cosine_d.shape[0]  
        

        # coefficient regularization to ensure plausible 3d faces
        creg_loss = self.opt.w_id * torch.sum(id_coef ** 2) +  \
               self.opt.w_exp * torch.sum(ex_coef ** 2) + \
               self.opt.w_tex * torch.sum(tex_coef ** 2)
        creg_loss = creg_loss / id_coef.shape[0]
        
        # gamma regularization to ensure a nearly-monochromatic light
        gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
        gamma_loss = torch.mean((gamma - gamma_mean) ** 2)

        # reflectance loss
        mask = self.BFM_model.skin_mask
        batch_size = tex_coef.shape[0]
        texture = torch.einsum('ij,aj->ai', tex_coef, self.BFM_model.texBase) + self.BFM_model.meantex
        texture = texture / 255.
        texture=texture.reshape([batch_size, -1, 3])
        mask = mask.reshape([1, mask.shape[1], 1])
        texture_mean = torch.sum(mask * texture, dim=1, keepdims=True) / torch.sum(mask)
        reflect_loss = torch.sum(((texture - texture_mean) * mask)**2) / (texture.shape[0] * torch.sum(mask))


        # total loss
        all_loss = self.w_color*img_loss + self.w_lm*lmk_loss + self.w_reg*creg_loss +\
            self.w_feat*feat_loss + self.w_gamma*gamma_loss + self.w_reflc*reflect_loss
        return all_loss, img_loss, lmk_loss, creg_loss, feat_loss, gamma_loss, reflect_loss, recon_img
