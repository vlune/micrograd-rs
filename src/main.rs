mod engine;
// mod nn;

use engine::Value;

fn main() {
    let x = Value::new(2.0);
    let y = Value::new(3.0);
    let z = &(&x * &y) + &x;

    z.backward();
    println!("z = {}", z.data());
    println!("dz/dx = {}", x.grad());
    println!("dz/dy = {}", y.grad());
}
