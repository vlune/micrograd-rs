mod engine;
mod nn;

use engine::Value;
use nn::MLP;

fn main() {
    // let x = Value::new(2.0);
    // let y = Value::new(3.0);
    // let z = &(&x * &y) + &x;

    // z.backward();
    // println!("z = {}", z.data());
    // println!("dz/dx = {}", x.grad());
    // println!("dz/dy = {}", y.grad());
    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let ys = [1.0, -1.0, -1.0, 1.0];

    let model = MLP::new(3, &[4, 4, 1]);

    for k in 0..100 {
        let mut loss = Value::new(0.0);
        for (x, y) in xs.iter().zip(&ys) {
            let input: Vec<Value> = x.iter().map(|&v| Value::new(v)).collect();
            let pred = model.forward(&input).remove(0);
            let diff = &pred - &Value::new(*y);
            loss = &loss + &diff.pow(2.0);
        }
        let n = xs.len() as f64;
        let loss = &loss * &Value::new(1.0 / n);
        model.zero_grad();
        loss.backward();

        model.step(0.01);

        if k % 10 == 0 {
            println!("step {k:3} loss = {:.6}", loss.data());
        }
    }
    println!("\n--- predictions ---");
    for (x, y) in xs.iter().zip(&ys) {
        let inputs: Vec<Value> = x.iter().map(|&v| Value::new(v)).collect();
        let pred = model.forward(&inputs).remove(0);
        println!("target = {:5.1}   pred = {:8.4}", y, pred.data());
    }
}
