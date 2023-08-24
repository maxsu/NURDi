from matplotlib import pyplot
import numpy
from pytorch3d.loss import chamfer_distance
import skimage
import torch
from tqdm import tqdm

from torch_nurbs_eval.curve_eval import CurveEval

torch.manual_seed(0)

def im_io(filepath):
    image = skimage.io.imread(filepath).astype(bool).astype(float)
    return img2pointcloud(image)


def img2pointcloud(image):
    point_cloud = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 1.0:
                boundary = 0
                if i < image.shape[0] - 1:
                    if image[i+1,j] == 0:
                        boundary = 1
                if j < image.shape[1] - 1:
                    if image[i,j+1] == 0:
                        boundary = 1
                if i > 0:
                    if image[i-1,j] == 0:
                        boundary = 1
                if j > 0:
                    if image[i,j-1] == 0:
                        boundary = 1
                if i < image.shape[0] - 1  and j < image.shape[1] - 1:
                    if image[i+1,j+1] == 0:
                        boundary = 1
                if i < image.shape[0] - 1  and j > 0:
                    if image[i+1,j-1] == 0:
                        boundary = 1
                if i > 0 and j < image.shape[1] - 1:
                    if image[i-1,j+1] == 0:
                        boundary = 1
                if i > 0 and j > 0:
                    if image[i-1,j-1] == 0:
                        boundary = 1
                if boundary == 1:
                    point_cloud.append([i+0.5,j+0.5])

    point_cloud = numpy.array(point_cloud)
    return point_cloud


def all_plot(ctrlpts,degree,predicted, target):
    ctrlpts = ctrlpts[0,:,:].tolist()
    predicted = predicted[:,:].tolist()
    target = target[0,:,:].tolist()



    pts = numpy.array(ctrlpts)
    pyplot.plot(pts[:, 0], pts[:, 1], color="blue", linestyle='-.', marker='o', label='Control Points',linewidth=0.5,markersize=2)

    pts = numpy.array(target)
    pyplot.scatter(pts[:,0],pts[:,1],marker=",",cmap = "Rdpu")

    pts = numpy.array(predicted)
    pyplot.scatter(pts[:,0],pts[:,1],marker=",",cmap = "CMRmap")
    
    pyplot.legend(loc=3,fontsize="large")
    
    # Pad margins so that markers don't get clipped by the axes
    pyplot.margins(0.08)
    
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 14,
            }
    pyplot.xlabel("$x$", fontdict=font)
    pyplot.ylabel("$y = 2cos(x) + exp(-x) + 0.5sin(-5x)$", fontdict=font)


## Main Script
target_np = im_io('./skeletons/cat.png')

target = torch.from_numpy(target_np).unsqueeze(0).float().cuda()

print("target")
print(target.size())
num_eval_pts = target.size(1)

print(num_eval_pts)

print("reached here")

print(numpy.min(target_np[:,0]))

# Compute and print loss
num_ctrl_pts = 32
indices = numpy.linspace(0,target_np.shape[0]-1,num_ctrl_pts,dtype=int)
print(target_np.shape)
print(indices)
xy_pts = target_np[indices]
print(xy_pts)
w_cpts = numpy.linspace(0.01,1,num_ctrl_pts)
cpts = numpy.array([xy_pts[:,0],xy_pts[:,1]]).T

inp_ctrl_pts = torch.from_numpy(cpts).unsqueeze(0)

inp_ctrl_pts = torch.nn.Parameter(inp_ctrl_pts)
layer = CurveEval(num_ctrl_pts, dimension=2, p=3, out_dim=num_eval_pts)
opt = torch.optim.Adam(iter([inp_ctrl_pts]), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min')
pbar = tqdm(range(100000))
for i in pbar:
    opt.zero_grad()
    weights = torch.ones(1,num_ctrl_pts,1)
    out = layer(torch.cat((inp_ctrl_pts,weights),axis=-1).float().cuda())
    out = out.float()

    loss,_ = chamfer_distance(out, target)
    if i < 3000:
        curve_length = ((out[:,0:-1,:] - out[:,1:,:])**2).sum((1,2)).mean()
        loss += 0.1*curve_length
    loss.backward()
    opt.step()
    scheduler.step(loss)
    if (i+1)%1000 == 0:
        target_mpl = target.cpu().numpy().squeeze()
        predicted = out.detach().cpu().numpy().squeeze()
        print(target_mpl.shape)
        print(predicted.shape)
        all_plot(inp_ctrl_pts,10,predicted,target)
        pyplot.show()
       
    pbar.set_descriptn("Loss %s: %s" % (i+1, loss.item()))
