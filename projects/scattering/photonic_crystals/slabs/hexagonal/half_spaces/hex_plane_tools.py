from scipy.linalg import expm, norm
import numpy as np


def rot_mat(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

def rotate_vector(v, axis, theta):
    M = rot_mat(axis, theta)
    return np.tensordot(M,v,axes=([0],[1])).T #np.dot(M, v)

def rotate_around_z(v, theta):
    return rotate_vector(v, np.array([0.,0.,1.]), theta)

def is_odd(num):
    return num & 0x1

def is_inside_hexagon(x, y, d=None, x0=0., y0=0.):
    p_eps = 10.*np.finfo(float).eps
    if d is None:
        d = y.max() - y.min() + p_eps
    dx = np.abs(x - x0)/d
    dy = np.abs(y - y0)/d
    a = 0.25 * np.sqrt(3.0)
    return np.logical_and(dx <= a, a*dy + 0.25*dx <= 0.5*a)

def get_hex_plane(plane_idx, inradius, z_height, z_center, np_xy,
                  np_z):
    
    # We use 10* float machine precision to correct the ccordinates
    # to avoid leaving the computational domain due to precision
    # problems
    p_eps = 10.*np.finfo(float).eps
    
    ri = inradius # short for inradius
    rc = inradius/np.sqrt(3.)*2. # short for circumradius
    
    if np_z == 'auto':
        np_z = int(np.round(float(np_xy)/2./rc*z_height))
    
    # XY-plane (no hexagonal shape!)
    if plane_idx == 6:
        X = np.linspace(-ri+p_eps, ri-p_eps, np_xy)
        Y = np.linspace(-rc+p_eps, rc-p_eps, np_xy)
        XY = np.meshgrid(X,Y)
        XYrs = np.concatenate((XY[0][..., np.newaxis], 
                               XY[1][..., np.newaxis]), 
                              axis=2)
        Z = np.ones((np_xy, np_xy, 1))*z_center
        pl = np.concatenate((XYrs, Z), axis=2)
        pl = pl.reshape(-1, pl.shape[-1])
        
        # Restrict to hexagon
        idx_hex = is_inside_hexagon(pl[:,0], pl[:,1])
        return pl[idx_hex]
    
    # Vertical planes
    elif plane_idx < 6:
        r = rc if is_odd(plane_idx) else ri
        r = r-p_eps
        xy_line = np.empty((np_xy,2))
        xy_line[:,0] = np.linspace(-r, r, np_xy)
        xy_line[:,1] = 0.
        z_points = np.linspace(0.+p_eps, z_height-p_eps, np_z)
        
        # Construct the plane
        plane = np.empty((np_xy*np_z, 3))
        for i, xy in enumerate(xy_line):
            for j, z in enumerate(z_points):
                idx = i*np_z + j
                plane[idx, :2] = xy
                plane[idx, 2] = z
        
        # Rotate the plane
        return rotate_around_z(plane, plane_idx*np.pi/6.)
    else:
        raise ValueError('`plane_idx` must be in [0...6].')

def get_hex_planes_point_list(inradius, z_height, z_center, np_xy, np_z,
                              plane_indices=[0,1,2,3,6]):
    # Construct the desired planes
    planes = []
    for i in plane_indices:
        planes.append(get_hex_plane(i, inradius, z_height, z_center, 
                                    np_xy, np_z))
    
    # Flatten and save lengths
    lengths = [len(p) for p in planes]
    return np.vstack(planes), np.array(lengths)

def hex_planes_point_list_for_keys(keys, plane_indices=[0,1,2,3,6]):
    if not 'uol' in keys:
        keys['uol'] = 1.e-9
    inradius = keys['p'] * keys['uol'] /2.
    z_height = (keys['h'] + keys['h_sub'] + keys['h_sup']) * keys['uol']
    z_center = (keys['h_sub']+keys['h']/2.) * keys['uol']
    np_xy = keys['hex_np_xy']
    if not 'hex_np_z' in keys:
        np_z = 'auto'
    return get_hex_planes_point_list(inradius, z_height, z_center, np_xy, 
                                     np_z)

def plane_idx_iter(lengths_):
    """Yields the plane index plus lower index `idx_i` and upper index 
    `idx_f` of the point list representing this plane
    (i.e. pointlist[idx_i:idx_f]).
    
    """
    i = 0
    while i < len(lengths_):
        yield i, lengths_[:i].sum(), lengths_[:(i+1)].sum()
        i += 1
        
def plot_planes(pointlist, lengths):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = sns.color_palette('husl', len(lengths))
    for i, idx_i, idx_f in plane_idx_iter(lengths):
        pl = pointlist[idx_i:idx_f]
        ax.scatter(pl[:,0], pl[:,1], pl[:,2], s=10., c=colors[i],
                   label='plane {}'.format(i+1), linewidth=0.)
    _ = plt.legend(loc='upper left')
