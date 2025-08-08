import numpy as np

class Tensor:
    def __init__(self, data, children=()):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.dtype != np.float64:
            data = data.astype(np.float64)

        self.data = data
        self.grad = np.zeros_like(self.data)
        self._prev = set(children)
        self._backward = lambda:None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    @staticmethod
    def sum_to(input_arr, target):
        target_shape = target.shape
        if input_arr.shape == target_shape:
            return input_arr
        
        # 1. Align shapes to determine which axes were broadcast
        ndim_diff = input_arr.ndim - len(target_shape)
        aligned_target_shape = (1,) * ndim_diff + target_shape
        
        # 2. Identify all axes that were stretched (either by prepending or originally being 1)
        # The loop is sufficient to find all of them; no pre-population needed.
        axes_to_sum = []
        for i, (input_dim, target_dim) in enumerate(zip(input_arr.shape, aligned_target_shape)):
            if input_dim != target_dim:
                axes_to_sum.append(i)

        # 3. Sum over the unique identified axes at once
        summed_arr = np.sum(input_arr, axis=tuple(axes_to_sum))
        
        # 4. Reshape to final target shape
        return summed_arr.reshape(target_shape)

    def __add__(self, other):
        out = Tensor(self.data + other.data, children=(self, other))
        def _backward():
            self.grad += Tensor.sum_to(out.grad, self.grad)
            other.grad += Tensor.sum_to(out.grad, other.grad)
        out._backward = _backward
        return out    
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        return self + (-other) 
    def __neg__(self):
        return -1 * self
    
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, children=(self, other))        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other) ##!!!
        out = Tensor(self.data * other.data, children=(self, other))
        def _backward():
            self.grad += Tensor.sum_to(other.data * out.grad, self.grad)
            other.grad += Tensor.sum_to(self.data * out.grad, other.grad)
        out._backward = _backward
        return out
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent must be int or float"
        out = Tensor(self.data ** other, children=(self,)) ##!!!!
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad ##!!!
        out._backward = _backward
        return out
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, children=(self,))
        def _backward():
            self.grad += (1 - t ** 2) * out.grad ## !! knowing when to use @ when in scalar it's normally multiplication
        out._backward = _backward
        return out
    
    def mean(self):
        out = Tensor(self.data.mean(), children=(self,))
        def _backward():
            self.grad += (1 / self.data.size) * out.grad 
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort 
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        
        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=np.float64)

        for node in reversed(topo):
            node._backward()

class Parameter(Tensor):
    def __init__(self, data, children=()):
        super().__init__(data, children)

class Module:
    def __init__(self):
        pass 
    
    def __call__(self):
        raise NotImplementedError 

    def parameters(self):
        raise NotImplementedError 

class Sequential(Module):
    def __init__(self, layers):
        super().__init__() 
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

class Linear(Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.w = Parameter(np.random.randn(nin, nout))
        self.b = Parameter(np.random.randn(nout))
    
    def __call__(self, x):
        out = x @ self.w + self.b
        return out

    def parameters(self):
        return [self.w, self.b] # how to return them

class Tanh(Module):
    def __init__(self):
        super().__init__() 
    
    def __call__(self, x):
        out = x.tanh()
        return out
    
    def parameters(self):
        return []

class MSELoss(Module):
    def __init__(self):
        super().__init__() 
    
    def __call__(self, y, y_targets):
        out = (0.5 * (y - y_targets) ** 2).mean()
        return out

class SGD:
    def __init__(self, model_params, lr):
        self.model_params = model_params 
        self.lr = lr

    def step(self):
        for p in self.model_params:
            p.data -= self.lr * p.grad
    
    def zero_grad(self):
        for p in self.model_params:
            p.grad = np.zeros_like(p.data)

if __name__ == "__main__":

    # Simple Linear Regression 
    X = Tensor(np.array([[0], [1], [2]]))
    y = Tensor(np.array([[0], [1], [2]]))

    lr = 0.1
    epochs = 1000

    model = Sequential([
                        Linear(nin=1, nout=2),
                        Tanh(),
                        Linear(nin=2, nout=1)
                        ])

    optimizer = SGD(model.parameters(), lr=lr) 
    criterion = MSELoss()

    print(f"Initial Predictions: {model(X)}")

    # Training
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y) 
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch + 1}, Loss: {loss.data}")
    
    print(f"New Predictions: {model(X)}")