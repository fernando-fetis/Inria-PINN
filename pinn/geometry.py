from shapely.affinity import affine_transform
from shapely.geometry import Point
from shapely.ops import triangulate
from shapely.prepared import prep
import shapely
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

class Domain:

    def __init__(self, type, temporal_domain=None, **kwargs) -> None:
        self.type = type  
        self.temporal_domain = temporal_domain

        if type=='polygon':
            self.spatial_domain= PolygonMesh(kwargs)

        elif type=='ellipsis':
            self.spatial_domain = EllipsisMesh(kwargs)
            pass

        elif type=='interval':
          self.spatial_domain = IntervalMesh(kwargs)
        
        elif type=='rectangle':
            self.spatial_domain = RectangleMesh(kwargs)
            
        if type in ('polygon', 'ellipsis'):
          self.variable_names = ['x', 'y']
        elif type == 'interval':
          self.variable_names = ['x']
        else:
          self.variable_names = list(kwargs.keys())  # rectangle.
          
        if temporal_domain is not None:
          self.variable_names.append('t')

    
    def sample(self, k_samples, at_initial=False, on_boundary=False, requires_grad=True, device='cpu'):

        data, normals = self.spatial_domain.spatial_sample(k_samples, on_boundary, requires_grad)

        if self.temporal_domain:
          for key in data.keys():
              if at_initial:
                  data[key]['t'] = torch.empty(len(data[key][self.variable_names[0]])).fill_(self.temporal_domain[0]).requires_grad_(requires_grad)
            
              else:
                  data[key]['t'] = torch.empty(len(data[key][self.variable_names[0]])).uniform_(*self.temporal_domain).requires_grad_(requires_grad)
        
        return (data, normals)


class SpatialGeometry:

  def __init__(self) -> None:
    
    self.interior_subdomains = {}
    self.boundary_subdomains = {}
    self.normals = {}
    self.samples = {}

    
  def add_interior_subdomain(self, subdomain, name=""):
    keys = self.interior_subdomains.keys()
    if name=="":
      name = "int"+str(len(keys)+1)
    elif name=="res":
      name = "int"+str(len(keys)+1)
      print("'res' is not a valid name for subdomain, instead using "+name)

    self.interior_subdomains[name] = subdomain
  

  def add_boundary_subdomain(self, subdomain, name=""):
    keys = self.boundary_subdomains.keys()
    if name=="":
      name = "int"+str(len(keys)+1)
    elif name=="res":
      name = "int"+str(len(keys)+1)
      print("'res' is not a valid name for subdomain, instead using "+name)

    self.boundary_subdomains[name] = subdomain


  def spatial_sample(self, k_samples, on_boundary=False, requires_grad=True):
    
    if on_boundary:
        points, normals = self._sample_boundary(k_samples, requires_grad)
    
    else:
        points, normals = self._sample_interior(k_samples, requires_grad)
    
    spatial_data = {key: {'x': points[key][0], 'y': points[key][1]}
                    for key in points.keys()}

    return spatial_data, normals


  def plot_samples(self):

      if self.samples == {}:
        print("No samples to plot. Generate samples first.")

      else:
        if 'interior' in self.samples.keys():
          for key in self.samples['interior'].keys():
            plt.plot(*self.samples['interior'][key], '.', label=key)

        if 'boundary' in self.samples.keys(): 
          for key in self.samples['boundary'].keys():
            plt.plot(*self.samples['boundary'][key], '.', label=key)

        plt.legend()


  def plot(self):
    x = np.array(self.geometry.exterior.coords.xy[0])
    y = np.array(self.geometry.exterior.coords.xy[1])

    plt.fill(x,y, facecolor='lightblue', edgecolor='blue')
    plt.show()


  def grid(self, step):

    latmin, lonmin, latmax, lonmax = self.geometry.bounds
    prep_polygon = prep(self.geometry)

    points = []
    for lat in np.arange(latmin, latmax, step):
        for lon in np.arange(lonmin, lonmax, step):
            points.append(Point((round(lat,4), round(lon,4))))

    valid_points = []
    valid_points.extend(filter(prep_polygon.contains, points))

    grid_coords = torch.tensor([p.coords for p in valid_points]).reshape(-1, 2)

    x, y = grid_coords.T

    return x, y


class PolygonMesh(SpatialGeometry):

  def __init__(self, kwargs) -> None:
    super().__init__()

    self.vertices = kwargs['vertices']
    P = shapely.geometry.Polygon(self.vertices)

    self.geometry = shapely.geometry.polygon.orient(P)
    self.edges = self._get_edges()


  def _get_edges(self):
    edges = []
    N_edges = len(self.geometry.boundary.coords)

    for i in range(N_edges-1):
      (x0, y0), (x1, y1) = self.geometry.boundary.coords[i], self.geometry.boundary.coords[i+1]
      edges.append([(x0, y0), (x1, y1)])
    
    return edges


  def _sample_interior(self, k_samples, requires_grad=True):

    self.samples['interior'] = {}

    i_points = {key: [] for key in list(self.interior_subdomains.keys()) + ['res']}

    T = triangulate(self.geometry)
    triangs = [t for t in T if self.geometry.contains(t)]

    areas = []
    transforms = []
    for t in triangs:
        areas.append(t.area)
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        transforms.append([x1 - x0, x2 - x0, y1 - y0, y2 - y0, x0, y0])

    for transform in random.choices(transforms, weights=areas, k=k_samples):
        x, y = np.random.random(2)
        if x + y > 1:
          x, y = 1 - x, 1 - y 
          p = Point(x, y)
        else:
          p = Point(x, y)

        t_point = affine_transform(p, transform)
          
        xn, yn = t_point.coords.xy[0][0], t_point.coords.xy[1][0]
        bound = 'res'

        for key in self.interior_subdomains.keys():
          if self.interior_subdomains[key](xn, yn):
            bound = key

        i_points[bound].append(t_point)

    for key in self.interior_subdomains.keys():
      points = i_points[key]
      torch_points = torch.tensor([p.coords for p in points], requires_grad=requires_grad).reshape(-1, 2).T
      self.samples['interior'][key] = torch_points

    torch_points = torch.tensor([p.coords for p in i_points['res']], requires_grad=requires_grad).reshape(-1, 2).T
    self.samples['interior']['res'] = torch_points

    return self.samples['interior'], None


  def _sample_boundary(self, k_samples, requires_grad=True):

    lengths, edges = [], []
    N_edges = len(self.geometry.boundary.coords)

    b_points = {key: [] for key in list(self.boundary_subdomains.keys()) + ['res']}

    for i in range(N_edges-1):
      (x0, y0), (x1, y1) = self.geometry.boundary.coords[i], self.geometry.boundary.coords[i+1]
      lengths.append(np.sqrt((x1-x0)**2 + (y1-y0)**2))
      edges.append([(x0, y0), (x1, y1)])

    for edge in random.choices(edges, weights=lengths, k=k_samples):
      lambd = np.random.random()
      (x0, y0), (x1, y1) = edge
      xn, yn = (1-lambd)*x0 + lambd*x1, (1-lambd)*y0 + lambd*y1

      bound = 'res'

      for key in self.boundary_subdomains.keys():
        if self.boundary_subdomains[key](xn, yn):
          bound = key

      b_points[bound].append(Point(xn, yn))

    self.samples['boundary'] = {}

    for key in list(self.boundary_subdomains.keys()) + ['res'] :

      torch_points = torch.tensor([p.coords for p in b_points[key]], requires_grad=requires_grad).reshape(-1, 2).T
      self.samples['boundary'][key] = torch_points
      self.normals[key] = self._normal_vector(torch_points[0], torch_points[1])

    return self.samples['boundary'], self.normals

  
  def _normal_vector(self, x, y, tol=1e-3):
    E = self.edges
    N = len(E)

    normals = torch.zeros((len(x), 2))

    for i in range(len(x)):

      con_edge = torch.zeros(2)
      xi, yi = x[i], y[i]

      for j in range(N):
        edge = shapely.geometry.LineString(E[j])
        buf = edge.buffer(tol)
        if buf.contains(Point(xi.item(), yi.item())):
          con_edge = torch.tensor(E[j][1]) - torch.tensor(E[j][0])

      if torch.dot(con_edge, con_edge)==0:
        print("Point not in boundary or Point it's a vertex")
        n = torch.zeros(2)

      elif con_edge[0].item() == 0:
        n = torch.tensor([1., 0.])
        n *= torch.sign(n[0]*con_edge[1] - n[1]*con_edge[0])

      else:
        e1 = torch.tensor([0., 1.])
        n = e1 - torch.dot(e1, con_edge)/torch.dot(con_edge, con_edge) * con_edge 
        n *= torch.sign(n[0]*con_edge[1] - n[1]*con_edge[0])/torch.linalg.norm(n)

      normals[i] = n

    return normals.reshape(2, len(x))


class IntervalMesh(SpatialGeometry):

  def __init__(self, kwargs) -> None:
     
     super().__init__()

     a, b = kwargs['a'], kwargs['b']
     self.spatial_domain = torch.tensor([a, b])


  def _sample_interior(self, k_samples, requires_grad=True):

    total = torch.empty(k_samples).uniform_(*self.spatial_domain).requires_grad_(requires_grad)
    samples = {}

    for key in self.interior_subdomains.keys():
      samples[key] = total[self.interior_subdomains[key](total)].copy()
      total = total[~self.interior_subdomains[key](total)].copy()

    samples['res'] = total

    return samples, None


  def _sample_boundary(self, k_samples, requires_grad=True):

    a, b = self.spatial_domain.numpy()

    choice = torch.tensor(random.choices([[a, -1], [b, 1]], [0.5, 0.5], k=k_samples), requires_grad=requires_grad)
    points, vectors = choice[:,0], choice[:,1]
    samples, normals = {}, {}

    for key in self.boundary_subdomains.keys():
      samples[key] = points[self.boundary_subdomains[key](points)].copy()
      normals[key] = vectors[self.boundary_subdomains[key](points)].copy()
      points = points[~self.boundary_subdomains[key](points)].copy()
      vectors = vectors[~self.boundary_subdomains[key](points)].copy()

    samples['res'] = points
    normals['res'] = vectors

    return samples, normals


  def spatial_sample(self, k_samples, on_boundary=False, requires_grad=True):
    
    if on_boundary:
        points, normals = self._sample_boundary(k_samples, requires_grad)
    
    else:
        points, normals = self._sample_interior(k_samples, requires_grad)
    
    spatial_data = {key: {'x': points[key]} for key in points.keys()}

    return spatial_data, normals

  
  def plot(self, step=0.01):

    x_axis = torch.arange(*self.spatial_domain(), step)

    plt.plot(x_axis, torch.zeros_like(x_axis))

  
  def grid(self, step):
    
    return torch.arange(*self.spatial_domain, step)


class EllipsisMesh(SpatialGeometry):
  def __init__(self, kwargs) -> None:
    super().__init__()

    center, x_scale, y_scale, angle = kwargs['center'], kwargs['x_scale'], kwargs['y_scale'], kwargs['angle']

    self.center = center
    self.x_scale = x_scale
    self.y_scale = y_scale 
    self.angle = torch.tensor(angle)

    self.rot_transform = torch.tensor([[torch.cos(self.angle), -torch.sin(self.angle)],
                                       [torch.sin(self.angle), torch.cos(self.angle)]])

    circ = shapely.geometry.Point(center).buffer(1)
    ell  = shapely.affinity.scale(circ, x_scale, y_scale)
    domain = shapely.affinity.rotate(ell, 90-angle)

    self.geometry = domain 

  def _sample_interior(self, k_samples):
    
    self.samples['interior'] = {}

    i_points = {key: [] for key in list(self.interior_subdomains.keys()) + ['res']}

    for _ in range(k_samples):

      r, theta = torch.rand(1), 2*np.pi*torch.rand(1)
      a, b = self.x_scale, self.y_scale
      point = torch.tensor([a*r*torch.cos(theta), b*r*torch.sin(theta)])
      rot_point = torch.matmul(self.rot_transform, point)

      bound = 'res'
      xn, yn = rot_point[0].item(), rot_point[1].item()

      for key in self.interior_subdomains.keys():
        if self.interior_subdomains[key](xn, yn):
          if len(i_points[key]) < k_samples[key]:
            bound = key
          else:
            bound = ''

        if bound != '':
          i_points[bound].append(rot_point)

    self.samples['interior'] = i_points


  def _sample_boundary(self, k_samples):

    self.samples['boundary'] = {}

    b_points = {key: [] for key in list(self.boundary_subdomains.keys()) + ['res']}

    for _ in range(k_samples):

      theta = 2*np.pi*torch.rand(1)
      a, b = self.x_scale, self.y_scale
      point = torch.tensor([a*torch.cos(theta), b*torch.sin(theta)])
      rot_point = torch.matmul(self.rot_transform, point)

      bound = 'res'
      xn, yn = rot_point[0].item(), rot_point[1].item()

      for key in self.boundary_subdomains.keys():
        if self.boundary_subdomains[key](xn, yn):
          if len(b_points[key]) < k_samples[key]:
            bound = key
          else:
            bound = ''

        if bound != '':
          b_points[bound].append(rot_point)

    torch_points = b_points.T
    self.samples['boundary'] = torch_points
    self.normals['res'] = self._normal_vector(torch_points[0], torch_points[1])

    return self.samples, self.normals


class RectangleMesh:

    def __init__(self, kwargs) -> None:

        self.spatial_domain = kwargs
        

    def _sample_interior(self, k_samples, requires_grad=True):

      b_points = {key: [] for key in list(self.boundary_subdomains.keys()) + ['res']}