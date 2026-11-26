use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Clone, Copy)]
enum Op {
    Leaf,
    Add,
    Mul,
    Pow(f64),
    Relu,
}

struct Inner {
    data: f64,
    grad: f64,
    op: Op,
    prev: Vec<Value>,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<Inner>>);

impl Value {
    pub fn new(data: f64) -> Self {
        Self::from_op(data, Op::Leaf, vec![])
    }
    fn from_op(data: f64, op: Op, prev: Vec<Value>) -> Self {
        Value(Rc::new(RefCell::new(Inner {
            data,
            grad: 0.0,
            op,
            prev,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    pub fn pow(&self, n: f64) -> Value {
        Value::from_op(self.data().powf(n), Op::Pow(n), vec![self.clone()])
    }

    pub fn relu(&self) -> Value {
        let d = self.data();
        Value::from_op(if d > 0.0 { d } else { 0.0 }, Op::Relu, vec![self.clone()])
    }
    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }
    pub fn sgd_step(&self, lr: f64) {
        let mut inner = self.0.borrow_mut();
        inner.data -= lr * inner.grad;
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Value {
        Value::from_op(
            self.data() + rhs.data(),
            Op::Add,
            vec![self.clone(), rhs.clone()],
        )
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: &Value) -> Value {
        Value::from_op(
            self.data() * rhs.data(),
            Op::Mul,
            vec![self.clone(), rhs.clone()],
        )
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * &Value::new(-1.0)
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: &Value) -> Value {
        self + &(-rhs)
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: &Value) -> Value {
        self * &rhs.pow(-1.0)
    }
}

impl Value {
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<*const RefCell<Inner>> = HashSet::new();

        fn build(v: &Value, visited: &mut HashSet<*const RefCell<Inner>>, topo: &mut Vec<Value>) {
            if visited.insert(Rc::as_ptr(&v.0)) {
                for child in v.0.borrow().prev.iter() {
                    build(child, visited, topo);
                }
                topo.push(v.clone());
            }
        }
        build(self, &mut visited, &mut topo);

        self.0.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            v.backward_step();
        }
    }

    fn backward_step(&self) {
        let inner = self.0.borrow();
        let out_grad = inner.grad;

        match inner.op {
            Op::Leaf => {}
            Op::Add => {
                for p in &inner.prev {
                    p.0.borrow_mut().grad += out_grad;
                }
            }
            Op::Mul => {
                let (a, b) = (&inner.prev[0], &inner.prev[1]);
                let (ad, bd) = (a.data(), b.data());

                a.0.borrow_mut().grad += bd * out_grad;
                b.0.borrow_mut().grad += ad * out_grad;
            }
            Op::Pow(n) => {
                let p = &inner.prev[0];
                let pd = p.data();
                p.0.borrow_mut().grad += n * pd.powf(n - 1.0) * out_grad;
            }
            Op::Relu => {
                let p = &inner.prev[0];
                let active = if inner.data > 0.0 { 1.0 } else { 0.0 };
                p.0.borrow_mut().grad += active * out_grad;
            }
        }
    }
}
