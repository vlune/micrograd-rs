use crate::engine::Value;
use rand::RngExt;

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::rng();
        let w = (0..nin)
            .map(|_| Value::new(rng.random_range(-1.0..1.0)))
            .collect();
        Neuron {
            w,
            b: Value::new(0.0),
            nonlin,
        }
    }
    pub fn forward(&self, x: &[Value]) -> Value {
        let act = self
            .w
            .iter()
            .zip(x)
            .fold(self.b.clone(), |acc, (wi, xi)| &acc + &(wi * xi));
        if self.nonlin { act.tanh() } else { act }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.w.clone();
        p.push(self.b.clone());
        p
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
        }
    }
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut sizes = vec![nin];
        sizes.extend_from_slice(nouts);
        let last = nouts.len() - 1;
        let layers = (0..nouts.len())
            .map(|i| Layer::new(sizes[i], sizes[i + 1], i != last))
            .collect();
        MLP { layers }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out = x.to_vec();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }
    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
    pub fn step(&self, lr: f64) {
        for p in self.parameters() {
            p.sgd_step(lr);
        }
    }
}
