import torch
from pinn.geometry import Domain

class BaseClass:
    
    def __init__(self):
        '''
        Initialize the base class of a generic PDE. The attributes defined below must be overridden
        according to the specific PDE problem. Each attribute indicates the following:
        
        - equation [callable]: receives input data (as dict of variables) and the network outputs. The output
            must be a tuple with each entry associated with a equation in the form T[u, data] = 0.
        - domain [Domain]: object generated by geometry. It contains method to sample points in the domain.
        - initial_condition [callable]: receives input data and the network outputs. It must return a tuple with
            each condition.
        - boundary_condition [callable]: it receives input data, the normal of each point in data and the output
            of the network. Normal directions are used in PDEs with Neumann conditions.
        - known_solution [callable]: method used in inverse problems for sampling known data. It only receives
            input data.
        - analytical_solution [callable]: used to compare the solution generated by the PINN with the analytical
            solution in case if it is known.
        
        Different examples can be found in the report.
        '''
        self.equation: callable
        self.domain: Domain
        self.initial_condition: callable = None
        self.boundary_condition: dict = None
        self.known_solution: callable = None
        self.analytical_solution: callable = None
    
    def d(self, f, variable):
        '''
        Method used in self.equation (or some constraint). It calculates the derivative of f with respect variable.
        '''
        return torch.autograd.grad(f.sum(), variable, create_graph=True)[0]
    
    def d_n(self, f, data: dict, normal_batch: torch.Tensor):
        '''
        Method usually used in self.boundary_condition when there is Neumann conditions. The parameters are
        the following:
        
        - data [dict]: dictionary of points separated by variables.
        - normal_batch [Tensor]: contains the normal directions of each point in data.
        '''
        n_points = normal_batch.shape[1]
        normal_derivatives = torch.zeros(n_points)
        spatial_data = [key for key in data.keys() if key != 't']
        
        for n_var, var in enumerate(spatial_data):
            normal_derivatives += self.d(f, data[var]) * normal_batch[n_var]

        return normal_derivatives