import numpy as np
import scipy.io as sio
import argparse
from camera import Camera
from plotting import *


# A very simple, but useful method to take the difference between the
# first and second element (usually for 2D vectors)
def diff(x):
    return x[1] - x[0]


'''
FORM_INITIAL_VOXELS  create a basic grid of voxels ready for carving

Arguments:
    xlim - The limits of the x dimension given as [xmin xmax]

    ylim - The limits of the y dimension given as [ymin ymax]

    zlim - The limits of the z dimension given as [zmin zmax]

    num_voxels - The approximate number of voxels we desire in our grid

Returns:
    voxels - An ndarray of size (N, 3) where N is approximately equal the 
        num_voxels of voxel locations.

    voxel_size - The distance between the locations of adjacent voxels
        (a voxel is a cube)

Our initial voxels will create a rectangular prism defined by the x,y,z
limits. Each voxel will be a cube, so you'll have to compute the
approximate side-length (voxel_size) of these cubes, as well as how many
cubes you need to place in each dimension to get around the desired
number of voxel. This can be accomplished by first finding the total volume of
the voxel grid and dividing by the number of desired voxels. This will give an
approximate volume for each cubic voxel, which you can then use to find the 
side-length. The final "voxels" output should be a ndarray where every row is
the location of a voxel in 3D space.
'''
def form_initial_voxels(xlim, ylim, zlim, num_voxels):
    # TODO: Implement this method!
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]
    total_volume = x_range * y_range * z_range
    cubic_volume = total_volume / num_voxels
    voxel_size = np.cbrt(cubic_volume)
    n_x, n_y, n_z = int(x_range / voxel_size), int(y_range / voxel_size), int(z_range / voxel_size)
    voxels = np.ones((n_x * n_y * n_z, 3), dtype=np.float64) * np.array([[xlim[0], ylim[0], zlim[0]]])
    x, y, z = np.arange(n_x), np.arange(n_y), np.arange(n_z)
    # Intuition: voxels should be something like:
    # [xmin, ymin, zmin]
    # [xmin + voxel_size, ymin, zmin]
    # ...
    # [xmin + (n_x-1)*voxel_size, ymin, zmin]
    # [xmin, ymin + voxel_size, zmin]
    # ...
    # So let voxels = ones * [xmin, ymin, zmin] + idx_matrix * voxel_size
    # where idx_matrix can be formed by np.repeat() and np.tile()
    # np.repeat([1, 2, 3], 2) => [1, 1, 2, 2, 3, 3]
    # np.tile([1, 2, 3], 2) => [1, 2, 3, 1, 2, 3] 
    x = np.tile(x, n_y * n_z).reshape(-1, 1)
    y = np.tile(np.repeat(y, n_x), n_z).reshape(-1, 1)
    z = np.repeat(z, n_x * n_y).reshape(-1, 1)
    voxels += np.c_[x, y, z] * voxel_size
    return voxels, voxel_size

'''
GET_VOXEL_BOUNDS: Gives a nice bounding box in which the object will be carved
from. We feed these x/y/z limits into the construction of the inital voxel
cuboid. 

Arguments:
    cameras - The given data, which stores all the information
        associated with each camera (P, image, silhouettes, etc.)

    estimate_better_bounds - a flag that simply tells us whether to set tighter
        bounds. We can carve based on the silhouette we use.

    num_voxels - If estimating a better bound, the number of voxels needed for
        a quick carving.

Returns:
    xlim - The limits of the x dimension given as [xmin xmax]

    ylim - The limits of the y dimension given as [ymin ymax]

    zlim - The limits of the z dimension given as [zmin zmax]

The current method is to simply use the camera locations as the bounds. In the
section underneath the TODO, please implement a method to find tigther bounds:
One such approach would be to do a quick carving of the object on a grid with 
very few voxels. From this coarse carving, we can determine tighter bounds. Of
course, these bounds may be too strict, so we should have a buffer of one 
voxel_size around the carved object. 
'''
def get_voxel_bounds(cameras, estimate_better_bounds = False, num_voxels = 4000):
    camera_positions = np.vstack([c.T for c in cameras])
    xlim = [camera_positions[:,0].min(), camera_positions[:,0].max()]
    ylim = [camera_positions[:,1].min(), camera_positions[:,1].max()]
    zlim = [camera_positions[:,2].min(), camera_positions[:,2].max()]

    # For the zlim we need to see where each camera is looking. 
    camera_range = 0.6 * np.sqrt(diff( xlim )**2 + diff( ylim )**2)
    for c in cameras:
        viewpoint = c.T - camera_range * c.get_camera_direction()
        zlim[0] = min( zlim[0], viewpoint[2] )
        zlim[1] = max( zlim[1], viewpoint[2] )

    # Move the limits in a bit since the object must be inside the circle
    xlim = xlim + diff(xlim) / 4 * np.array([1, -1])
    ylim = ylim + diff(ylim) / 4 * np.array([1, -1])

    if estimate_better_bounds:
        # TODO: Implement this method!
        voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)
        for c in cameras:
            voxels = carve(voxels, c)
        low = np.min(voxels, axis=0) - voxel_size
        high = np.max(voxels, axis=0) + voxel_size
        xlim[0], ylim[0], zlim[0] = low[0], low[1], low[2]
        xlim[1], ylim[1], zlim[1] = high[0], high[1], high[2]
    return xlim, ylim, zlim
    

'''
CARVE: carves away voxels that are not inside the silhouette contained in 
    the view of the camera. The resulting voxel array is returned.

Arguments:
    voxels - an Nx3 matrix where each row is the location of a cubic voxel

    camera - The camera we are using to carve the voxels with. Useful data
        stored in here are the "silhouette" matrix, "image", and the
        projection matrix "P". 

Returns:
    voxels - a subset of the argument passed that are inside the silhouette
'''
def carve(voxels, camera):
    # TODO: Implement this method!
    sil = camera.silhouette
    P = camera.P
    idx = np.nonzero(sil)
    x_min, x_max = np.min(idx[1]), np.max(idx[1])
    y_min, y_max = np.min(idx[0]), np.max(idx[0])
    # map 3d point to 2d point
    one = np.ones((voxels.shape[0], 1))
    point_3d = np.c_[voxels, one]
    point_2d = P.dot(point_3d.T).T
    point_2d /= point_2d[:, 2:3]
    point_2d = point_2d[:, :2]
    point_2d = point_2d.astype(int)
    mask = (point_2d[:, 0] >= x_min) & (point_2d[:, 0] <= x_max) & (point_2d[:, 1] >= y_min) & (point_2d[:, 1] <= y_max)
    subidx = np.argwhere(mask == True).flatten()
    subpoint = point_2d[subidx, :]
    # validate if every subpoint is in the silhouette
    subidx = subidx[sil[subpoint[:, 1], subpoint[:, 0]] == True]
    return voxels[subidx, :]




'''
ESTIMATE_SILHOUETTE: Uses a very naive and color-specific heuristic to generate
the silhouette of an object

Arguments:
    im - The image containing a known object. An ndarray of size (H, W, C).

Returns:
    silhouette - An ndarray of size (H, W), where each pixel location is 0 or 1.
        If the (i,j) value is 0, then that pixel location in the original image 
        does not correspond to the object. If the (i,j) value is 1, then that
        that pixel location in the original image does correspond to the object.
'''
def estimate_silhouette(im):
    return np.logical_and(im[:,:,0] > im[:,:,2], im[:,:,0] > im[:,:,1] )


if __name__ == '__main__':
    estimate_better_bounds = True
    use_true_silhouette = True
    frames = sio.loadmat('./ps3_code/space_carving/frames.mat')['frames'][0]
    cameras = [Camera(x) for x in frames]

    # Generate the silhouettes based on a color heuristic
    if not use_true_silhouette:
        for i, c in enumerate(cameras):
            c.true_silhouette = c.silhouette
            c.silhouette = estimate_silhouette(c.image)
            if i == 0:
                plt.figure()
                plt.subplot(121)
                plt.imshow(c.true_silhouette, cmap = 'gray')
                plt.title('True Silhouette')
                plt.subplot(122)
                plt.imshow(c.silhouette, cmap = 'gray')
                plt.title('Estimated Silhouette')
                plt.show()

    # Generate the voxel grid
    # You can reduce the number of voxels for faster debugging, but
    # make sure you use the full amount for your final solution
    num_voxels = 6e6
    xlim, ylim, zlim = get_voxel_bounds(cameras, estimate_better_bounds)

    # This part is simply to test forming the initial voxel grid
    # voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, 4000)
    # plot_surface(voxels)
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)

    # Test the initial carving
    voxels = carve(voxels, cameras[0])
    if use_true_silhouette:
        plot_surface(voxels)
    # Result after all carvings
    for c in cameras:
        voxels = carve(voxels, c)
    plot_surface(voxels, voxel_size)