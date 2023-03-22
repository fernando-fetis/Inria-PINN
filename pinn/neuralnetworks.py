import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.tri as tri
import shapely
import numpy as np
import math

class FullyConnected(nn.Module):
    '''Creates a fully connected NN used as a PINN.'''

    def __init__(self,
                 n_variables: int,
                 n_output: int,
                 n_layers: int,
                 wide: int, 
                 activation: nn.Module = nn.Tanh(),
                 dropout: float = 0.0,
                 bn: bool = False):
        
        '''
        Args:

            n_variables: Number of variables of the equation solutiones, must to be an integer.
            Consider that n_variables=1 corresponds to an ODE.

            n_outputs: Number of solutions that the net has to return, also considering the parameters
            that the network will aproximate in the inverse problem. A simple PDE problem like a Poisson
            equation will have one output. A system of PDE will have more than one output. For example in 
            in the problem -div(sigma*grad(u)) = f, with boundary conditions, the n_outputs has to be setted
            in 2 if the parameter sigma needs to be estimated like the unknown solution u.

            n_layers: Number of layers that the net will have.

            wide: Number of neurons per layer.

            activation: Function of nn.Module of Pytorch that acts as the activation funcion on each layer. To see
            all activation functions see:
            https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

            Defaults to nn.Tanh().

            dropout: Probability parameter of Dropout on each layer, has to be 0 <= p <= 1.

            bn: Indicates if Batch Normalization will be applied or not. False indicates not. Defaults False.
        '''

        super().__init__()
        
        # Internal attributes:
        self.conf_interval = None
        self.confidence = None
        self.n_output = n_output
        
        # Modules:
        self.inner_layers = nn.ModuleList([nn.Linear(n_variables if n == 0 else wide, wide)
                                           for n in range(n_layers - 1)])
        self.inner_layers_bn = nn.ModuleList([nn.BatchNorm1d(wide) if bn else nn.Identity()
                                              for n in range(n_layers - 1)])
        self.last_layer = nn.Linear(wide, n_output)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        # Initialization:
        def init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
        
        if self.activation in (nn.Tanh(), nn.Sigmoid()):
            self.apply(init_weights)
    
    
    def forward(self, data: dict):
        '''Forward method of nn.Module.

        Args:
            data: Dictionary whose keys are the variables of equation solution, with values torch.tensor
            corresponding to points to evaluate the network.

        Examples:
            >>> data = {'x': torch.tensor([1., 2.]), 't': torch.tensor([0.5, 0.8])}
            >>> data = {'x': torch.tensor([1., 2.]), 'y': torch.tensor([2., 3.]), 't': torch.tensor([0.5, 0.8])}
        '''
        input = torch.stack(list(data.values()), axis=1)
        
        for n_layer in range(len(self.inner_layers)):
            input = self.activation(self.inner_layers[n_layer](input))
            input = self.inner_layers_bn[n_layer](input)
            input = self.dropout(input)
        input = self.last_layer(input)
        
        return tuple(input[:, eq] for eq in range(self.n_output))



    def condifence_interval(self, data: dict):
            '''Make a report of the confidence interval for a single point given as
            dictionary format equal as forward method format.
            Args:
                data: Dictionary whose keys are the variables of equation solution, with values torch.tensor
                corresponding to points to evaluate the network.
            Prints for each output in the net the confidence interval computed in the
            training epoch, if was indicated, as follows:
            Confidence of confidence setted in training, defaults 95% or 0.95
            Output number:
            Predicted value: net estimation
            Mean value: mean value of the epochs given to confidence interval calculation.
            Lower value: lower value of the confidence interval computed in training part.
            Upper value: upper value of the confidence interval computed in training part.
            Example (for a equation of dimension 2, with 2 equations and 2 outputs):
            >>> data = {'x': torch.tensor([1.]), 't': torch.tensor([.5])}
            >>> pinn.condifence_interval(data)
            >>> Confidence of 95.0%:
                Output number 1:
                Predicted value: 1.46025550365448
                Mean value: 1.4011306762695312
                Lower value: 1.1304807662963867
                Upper value: 1.6717805862426758
                Output number 2:
                Predicted value: 1.1432186365127563
                Mean value: 1.07851243019104
                Lower value: 0.8631584048271179
                Upper value: 1.293866515159607
            '''
            assert self.conf_interval is not None, 'First compute confidence interval.'
            input_inter = [torch.stack(list(data.values()), axis=1).numpy()[0], ]
            pred_value = self(data)
            n_eqs = len(pred_value)
            mean_value, lower_value, upper_value = self.conf_interval
            print("Confidence of "+str(self.confidence*100)+"% for:")
            print()
            for i in range(n_eqs):
                mv, lv, uv = mean_value[i], lower_value[i], upper_value[i]
                m, L, U = mv(input_inter), lv(input_inter), uv(input_inter)
                print("Output number "+str(i+1)+":")
                print()
                print(f'Predicted value: {pred_value[i].item()}')
                print(f'Mean value: {m[0]}')
                print(f'Lower value: {L[0]}')
                print(f'Upper value: {U[0]}')
                print()



    def plot_confidence(self, pde: object, step: float = 1e-2):
        '''Plots the confidence interval of the solution. Will be a plot of ncols=number of outputs 
        of the network.

        1D equations (ODEs): Plot the solution predicted by the net (best epoch) in orange, the 
        mean solution (of last epochs in confidence interval computation) in blue and the lower and upper
        interval like yellow fill between.
        2D equations: Plot the mean solution like a surface, and the lower and upper likes a surface with 
        little alpha for transparency.

        The grid of the plot will be equidistant with step equal to step.

        Args:

            pde: Object of PDE class, has to be the same equation that PINN aproximates to solve.
            float: Side length of grid generate to plot. Defaults to 1e-2.
        '''
        assert self.conf_interval is not None, 'First compute confidence interval.'

        self.eval()
        self.cpu()

        if len(pde.domain) == 2:

            data = [torch.arange(*domain, step) for domain in pde.domain.values()]
            grid_1, grid_2 = torch.meshgrid(*data, indexing='ij')
            
            var_names = list(pde.domain.keys())
            input = {var_names[0]: grid_1.flatten(), var_names[1]: grid_2.flatten()}
            u_pred = self(input)
            ncols = len(u_pred)
            
            M, L, U = self.conf_interval
            g1, g2 = grid_1.detach().numpy(), grid_2.detach().numpy()
                
            fig, ax = plt.subplots(nrows=1, ncols=ncols , figsize=(6*ncols, 5), subplot_kw={'projection': '3d'})
            
            for i in range(ncols):
                m, l, u = M[i], L[i], U[i]
                ax[i].plot_surface(g1, g2, m((g1, g2)), cmap = "Spectral")
                ax[i].plot_surface(g1, g2, l((g1, g2)), alpha = 0.2)
                ax[i].plot_surface(g1, g2, u((g1, g2)), alpha = 0.2)
                ax[i].set_xlabel(f'${var_names[0]}$')
                ax[i].set_ylabel(f'${var_names[1]}$')
                ax[i].set_title('u '+str(i+1))

            plt.show()


        elif len(pde.domain) == 1:

            var_name = list(pde.domain.keys())[0]
            dom_esp = torch.arange(*pde.domain[var_name], step)
            dom_det = dom_esp.detach()
            input = {var_name: dom_esp}
            u_pred = self(input)
            ncols = len(u_pred)

            M, L, U = self.conf_interval
            fig, ax = plt.subplots(nrows=1, ncols=ncols , figsize=(6*ncols, 5))

            if ncols == 1:
                m, l, u = M[0], L[0], U[0]
                ax.plot(dom_det, m(dom_det), label='Mean solution')
                ax.fill_between(dom_det, l(dom_det), u(dom_det), label='CI '+str(self.confidence*100)+'%', color='yellow', alpha=0.5)
                ax.plot(input[var_name], u_pred[0].detach().numpy(), label='PINN')

            else:
                for i in range(ncols):
                    m, l, u = M[i], L[i], U[i]
                    ax.plot(dom_det, m(dom_det), label='Mean solution')
                    ax.fill_between(dom_det, l(dom_det), u(dom_det), label='CI '+str(self.confidence*100)+'%', color='yellow', alpha=0.5)
                    ax.plot(input[var_name], u_pred[i].detach().numpy(), label='PINN')        
            
            plt.xlabel(var_name); plt.ylabel(f'$u({var_name})$')
            plt.title('PINN simulation '); plt.legend()
            plt.grid(alpha=0.3, linestyle='--')


    def plot_pinn(self,
                  pde: object,
                  step: float = 0.05,
                  plain_plot: bool = True,
                  ani_config: dict = {'filename': 'pinn.gif', 'temporal values': 50, 'fps': 15, 'dpi': 100}):
        
        '''
        Plots the function learned by the neural network. The args are:
        
        - pde [pde object]: contains all information about the PDE. Used to sample points in the domain and to compare with the
            analytical solution in case it is defined.
        - step [float]: step used in the linspace data for the simulaiton. A small step will take longer to simulate.
        - plain_plot [bool]: used when the PDE has 2 spatial variables. if True, the plot will be a 2D colormap. Else, the plot will
            be a 3D surface.
        - ani_config [dict]: used when the PDE is temporal. Indicates the parameters needed for generating the gif associated
            to the simulation.
        '''
        
        var_names = pde.domain.variable_names
        spatial_vars = len(var_names) if pde.domain.temporal_domain is None else len(var_names) - 1

        if spatial_vars > 2:
            raise Exception('The PDE must have at most 2 spatial variables to be plotted.')
        
        # Data generation for the simulation:
        
        elif spatial_vars == 2:
            x_plot, y_plot =pde.domain.spatial_domain.grid(step)
            data = {var_names[0]: x_plot, var_names[1]: y_plot}
            triang = tri.Triangulation(x_plot, y_plot)
            x_triang = x_plot[triang.triangles].mean(axis=1)
            y_triang = y_plot[triang.triangles].mean(axis=1)
            cond = shapely.contains_xy(pde.domain.spatial_domain.geometry, x_triang, y_triang)
            triang.set_mask(np.where(cond==1, 0, 1))
            data_plot = triang
            
        elif spatial_vars == 1:
            x_plot = pde.domain.spatial_domain.grid(step)
            data = {var_names[0]: x_plot}
            data_plot = x_plot
            
        # Number of plots needed: 
        if pde.analytical_solution is not None:
            ncols = 3
            titles = ('PINN prediction', 'Analytical solution', 'Absolute error')
        else:
            ncols = 1
            titles = ('PINN prediction', )
        nrows = self.n_output
        
        proj = None if plain_plot else '3d'
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols , figsize=(6*ncols, 5*nrows), squeeze=False, subplot_kw={'projection': proj})
        self.cpu()

        def _plot():
            '''
            Inner method for makes the plots. It is encapsulated in order to allow animations in temporal PDEs.
            '''
            
            # Data that will be plotted:
            u_pred = self(data)
            if pde.analytical_solution is not None:
                u_real = pde.analytical_solution(data)
                error = [abs(real - pred) for real, pred in zip(u_pred, u_real)]
                plots = (u_pred, u_real, error)
            else:
                plots = (u_pred, )
            
            # Plotting:
            for eq in range(nrows):
                for sim in range(ncols):
                    axis = ax[eq, sim]
                    axis.clear()  # needed for making animations.
                    if spatial_vars == 2 and plain_plot:
                        simulation = axis.tricontourf(data_plot, plots[sim][eq].detach(), cmap='Spectral', levels=30)
                        if not pde.domain.temporal_domain:  # colorbar has a bug when is used with FuncAnimation.
                            fig.colorbar(simulation)
                    elif spatial_vars == 2 and not plain_plot:
                        simulation = axis.plot_trisurf(data_plot, plots[sim][eq].detach(), cmap='Spectral')
                    else:  # only one spatial variable.
                        simulation = axis.plot(data_plot, plots[sim][eq].detach())
                        
                    axis.set_title(titles[sim], fontdict={'size': 16})
                    axis.set_xlabel(f'${var_names[0]}$')
                    axis.set_ylabel(f'${var_names[1]}$')

        if pde.domain.temporal_domain:
            def _animate(t):
                data['t'] = torch.empty_like(x_plot).fill_(t)
                _plot()
            print('Generating animation...')
            t_range = torch.linspace(*pde.domain.temporal_domain, ani_config['temporal values'])
            ani = FuncAnimation(fig, _animate, frames=t_range)
            ani.save(ani_config['filename'], fps=ani_config['fps'], dpi=ani_config['dpi'])
            print(f'Simulation saved as {ani_config["filename"]}.')
            plt.close()
        else:
            _plot()
            plt.show()
  
# Future work:

class _SelfAttention(nn.Module):
    
    def __init__(self, d_model, dk, dv):
        super().__init__()
        self.Q = nn.Linear(d_model, dk)
        self.K = nn.Linear(d_model, dk)
        self.V = nn.Linear(d_model, dv)
    
    def forward(self, x):
        dk = self.K.out_features
        gram_matrix = self.Q(x) @ self.K(x).transpose(0, 1)
        scores = nn.Softmax(dim=1)(gram_matrix / math.sqrt(dk))
        self_attention = scores @ self.V(x)
        return self_attention

class AttentionPINN(nn.Module):
  
  def __init__(self, d_model, d_attn_inner, d_attn_output, d_output):

    super().__init__()

    self.attn1 = _SelfAttention(d_model, d_attn_inner, d_attn_inner)
    self.attn2 = _SelfAttention(d_attn_inner, d_attn_inner, d_attn_output)
    self.fc = nn.Linear(d_attn_output, d_output)

  def forward(self, x, y):
    input = torch.cat([x, y], dim=1)
    input = nn.GELU()(self.attn1(input))
    input = nn.GELU()(self.attn2(input))
    input = self.fc(input)
    return input